"""
=============================================================================
TRAIN FINAL — Production submission pipeline
=============================================================================
Strategy:
  NORMAL insurers (A,B,C,D,F,G,H,J):
    - Quick 90/10 validation → pick best model + iteration count + weight
    - Retrain winner on 100% data with fixed iterations
    - Predict Block 2 & 3

  DIFFICULT insurers (E,I,K):
    - Split by coverage (mtpl, limited_casco, casco)
    - Outlier clipping at p99.5 + log transform
    - Same validation → retrain 100% per coverage slice
    - Route test rows to correct coverage model

Usage:
    1. First run: python clean_data.py
    2. Then run:  python train_final.py

Runtime: ~90-120 minutes
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys
import time
import json
import gc
import catboost
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLEAN_DIR = "data_cleaned"
OUTPUT_DIR = "submissions_final"
MODEL_DIR = "saved_models_final"

ALL_INSURERS = [
    "Insurer_A", "Insurer_B", "Insurer_C", "Insurer_D", "Insurer_E",
    "Insurer_F", "Insurer_G", "Insurer_H", "Insurer_I", "Insurer_J",
    "Insurer_K"
]

NORMAL_INSURERS = ["Insurer_A", "Insurer_B", "Insurer_C", "Insurer_D",
                   "Insurer_F", "Insurer_G", "Insurer_H", "Insurer_J"]

DIFFICULT_INSURERS = ["Insurer_E", "Insurer_I", "Insurer_K"]

COVERAGE_TYPES = ["mtpl", "limited_casco", "casco"]

OUTLIER_CLIP_PERCENTILE = 99.5
USE_LOG_TRANSFORM = True
MIN_ROWS_FOR_COVERAGE_SPLIT = 1000

# LightGBM
LGB_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
LGB_ROUNDS = 1000
LGB_EARLY_STOP = 150

# CatBoost
CB_PARAMS = {
    "loss_function": "MAE",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 10,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.8,
    "allow_writing_files": False,
    "verbose": 500,
    "random_seed": 42,
    "thread_count": -1,
    "task_type": "CPU",
}
CB_EARLY_STOP = 150


# =============================================================================
# DATA LOADING
# =============================================================================
def detect_separator(filepath):
    best_sep, best_cols = ",", 0
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=3, encoding="utf-8")
            if len(df.columns) > best_cols:
                best_cols = len(df.columns)
                best_sep = sep
        except:
            continue
    return best_sep


def load_cleaned_data():
    print("[1/8] LOADING CLEANED DATA...")
    files = {
        "train": os.path.join(CLEAN_DIR, "block1_train_cleaned.csv"),
        "test_b2": os.path.join(CLEAN_DIR, "block2_test_cleaned.csv"),
        "test_b3": os.path.join(CLEAN_DIR, "block3_test_cleaned.csv"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found! Run clean_data.py first.")
            sys.exit(1)

    sep = detect_separator(files["train"])

    print("  Loading training data...")
    train = pd.read_csv(files["train"], sep=sep, low_memory=False)
    train.columns = train.columns.str.strip().str.replace(" ", "_")
    print(f"    → {train.shape[0]:,} rows × {train.shape[1]} cols")

    print("  Loading block2 test...")
    test_b2 = pd.read_csv(files["test_b2"], sep=sep, low_memory=False)
    test_b2.columns = test_b2.columns.str.strip().str.replace(" ", "_")
    print(f"    → {test_b2.shape[0]:,} rows × {test_b2.shape[1]} cols")

    print("  Loading block3 test...")
    test_b3 = pd.read_csv(files["test_b3"], sep=sep, low_memory=False)
    test_b3.columns = test_b3.columns.str.strip().str.replace(" ", "_")
    print(f"    → {test_b3.shape[0]:,} rows × {test_b3.shape[1]} cols")

    return train, test_b2, test_b3


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    df = df.copy()
    today = pd.Timestamp("2026-03-28")

    # Driver age
    df["contractor_birthdate"] = pd.to_datetime(
        df["contractor_birthdate"], format="%d/%m/%Y", errors="coerce"
    )
    df["driver_age"] = (today - df["contractor_birthdate"]).dt.days / 365.25
    df["driver_age"] = df["driver_age"].clip(17, 100)
    df["is_young_driver"] = (df["driver_age"] < 25).astype(np.int8)
    df["is_senior_driver"] = (df["driver_age"] > 65).astype(np.int8)
    df["driver_age_bucket"] = pd.cut(
        df["driver_age"], bins=[0, 21, 25, 30, 40, 50, 60, 70, 100],
        labels=[0, 1, 2, 3, 4, 5, 6, 7]
    ).astype(float)

    # Vehicle dates
    for col in ["vehicle_first_registration_date", "vehicle_country_first_registration_date",
                "vehicle_last_registration_date", "vehicle_inspection_report_date",
                "vehicle_inspection_expiry_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            df[f"{col}_days_ago"] = (today - df[col]).dt.days

    if "vehicle_inspection_expiry_date" in df.columns:
        df["inspection_remaining_days"] = (df["vehicle_inspection_expiry_date"] - today).dt.days
        df["inspection_is_valid"] = (df["inspection_remaining_days"] > 0).astype(np.int8)

    if "vehicle_last_registration_date" in df.columns:
        df["days_since_last_registration"] = (today - df["vehicle_last_registration_date"]).dt.days

    # Vehicle derived
    v_age = df["vehicle_age"].fillna(0).clip(lower=0) + 1
    if "vehicle_value_new" in df.columns:
        df["vehicle_value_log"] = np.log1p(df["vehicle_value_new"].clip(lower=0))
        df["vehicle_value_per_age"] = df["vehicle_value_new"] / v_age

    if "vehicle_power" in df.columns:
        df["vehicle_power_log"] = np.log1p(df["vehicle_power"].clip(lower=0))
        if "vehicle_engine_size" in df.columns:
            df["power_per_cc"] = df["vehicle_power"] / df["vehicle_engine_size"].replace(0, np.nan)
        if "vehicle_net_weight" in df.columns:
            df["power_to_weight"] = df["vehicle_power"] / df["vehicle_net_weight"].replace(0, np.nan)

    dims = ["vehicle_length", "vehicle_width", "vehicle_height"]
    if all(d in df.columns for d in dims):
        df["vehicle_volume"] = df["vehicle_length"] * df["vehicle_width"] * df["vehicle_height"]
        df["vehicle_footprint"] = df["vehicle_length"] * df["vehicle_width"]

    if "vehicle_net_weight" in df.columns:
        weight = df["vehicle_net_weight"].replace(0, np.nan)
        if "vehicle_value_new" in df.columns:
            df["value_per_kg"] = df["vehicle_value_new"] / weight

    if "vehicle_planned_annual_mileage" in df.columns:
        df["mileage_log"] = np.log1p(df["vehicle_planned_annual_mileage"])
        df["is_high_mileage"] = (df["vehicle_planned_annual_mileage"] > 25000).astype(np.int8)

    # Coverage
    if "coverage" in df.columns:
        cov_map = {"mtpl": 0, "limited_casco": 1, "casco": 2}
        df["coverage_ordinal"] = df["coverage"].str.lower().str.strip().map(cov_map)

    # Claim-free years
    if "claim_free_years" in df.columns:
        df["has_claims_history"] = (df["claim_free_years"] < 0).astype(np.int8)
        df["claim_free_abs"] = df["claim_free_years"].abs()
        df["claim_risk_bucket"] = pd.cut(
            df["claim_free_years"], bins=[-999, -5, -1, 0, 3, 7, 15, 999],
            labels=[0, 1, 2, 3, 4, 5, 6]
        ).astype(float)
        df["estimated_claims"] = np.where(
            df["claim_free_years"] < 0, np.ceil((-df["claim_free_years"]) / 5), 0
        )

    # Deductible features
    ded_cols = [f"{ins}_deductible" for ins in ALL_INSURERS if f"{ins}_deductible" in df.columns]
    if len(ded_cols) > 0:
        ded_matrix = df[ded_cols]
        df["deductible_mean"] = ded_matrix.mean(axis=1)
        df["deductible_std"] = ded_matrix.std(axis=1)
        df["deductible_min"] = ded_matrix.min(axis=1)
        df["deductible_max"] = ded_matrix.max(axis=1)
        df["deductible_range"] = df["deductible_max"] - df["deductible_min"]
        df["deductible_nonmissing_count"] = ded_matrix.notna().sum(axis=1)
        for ins in ALL_INSURERS:
            ded_col = f"{ins}_deductible"
            if ded_col in df.columns:
                df[f"{ins}_ded_vs_mean"] = df[ded_col] - df["deductible_mean"]
                df[f"{ins}_ded_vs_median"] = df[ded_col] - ded_matrix.median(axis=1)

    # Booleans
    for col in ["is_driver_owner", "vehicle_is_imported",
                "vehicle_is_imported_within_last_12_months",
                "vehicle_can_be_registered", "vehicle_has_open_recall",
                "vehicle_is_marked_for_export"]:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.lower().str.strip().map(
                    {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categoricals → _cat
    for col in ["vehicle_maker", "vehicle_model", "vehicle_fuel_type",
                "vehicle_primary_color", "vehicle_odometer_verdict_code",
                "coverage", "payment_frequency", "province", "municipality"]:
        if col in df.columns:
            df[col + "_cat"] = df[col].astype(str).str.strip().str.lower()

    # Maker+model combo
    if "vehicle_maker" in df.columns and "vehicle_model" in df.columns:
        maker = df["vehicle_maker"].astype(str).str.strip().str.lower()
        model = df["vehicle_model"].astype(str).str.strip().str.lower()
        df["model_equals_maker"] = (maker == model).astype(np.int8)
        df["vehicle_maker_model_cat"] = maker + "_" + model

    # Frequency encoding
    for col in ["vehicle_maker", "vehicle_model", "municipality", "postal_code"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_frequency"] = df[col].map(freq)

    # Deductible / value ratios
    if "vehicle_value_new" in df.columns:
        val_safe = df["vehicle_value_new"].replace(0, np.nan)
        for ins in ALL_INSURERS:
            ded_col = f"{ins}_deductible"
            if ded_col in df.columns:
                df[f"{ins}_ded_to_value_ratio"] = df[ded_col] / val_safe

    # Interactions
    cov = df.get("coverage_ordinal", pd.Series(dtype=float))
    age = df.get("driver_age", pd.Series(dtype=float))
    val = df.get("vehicle_value_new", pd.Series(dtype=float))
    cfy = df.get("claim_free_years", pd.Series(dtype=float))
    pwr = df.get("vehicle_power", pd.Series(dtype=float))
    vage = df.get("vehicle_age", pd.Series(dtype=float))
    mile = df.get("vehicle_planned_annual_mileage", pd.Series(dtype=float))

    if len(cov) > 0 and len(age) > 0: df["coverage_x_driver_age"] = cov * age
    if len(cov) > 0 and len(val) > 0: df["coverage_x_vehicle_value"] = cov * val
    if len(cov) > 0 and len(cfy) > 0: df["coverage_x_claim_free"] = cov * cfy
    if len(cov) > 0 and len(pwr) > 0: df["coverage_x_power"] = cov * pwr
    if len(cov) > 0 and len(vage) > 0: df["coverage_x_vehicle_age"] = cov * vage
    if len(age) > 0 and len(cfy) > 0: df["age_x_claim_free"] = age * cfy
    if len(age) > 0 and len(pwr) > 0: df["age_x_power"] = age * pwr
    if len(age) > 0 and len(mile) > 0: df["age_x_mileage"] = age * mile
    if len(vage) > 0 and len(val) > 0: df["vehicle_age_x_value"] = vage * val
    if len(pwr) > 0 and len(vage) > 0: df["power_x_vehicle_age"] = pwr * vage
    if len(cfy) > 0 and len(val) > 0: df["claim_free_x_value"] = cfy * val
    if len(mile) > 0 and len(cov) > 0: df["mileage_x_coverage"] = mile * cov
    if len(age) > 0 and len(val) > 0: df["age_x_value"] = age * val

    return df


def get_feature_columns(df, insurer):
    exclude_exact = {
        "quote_id", "contractor_birthdate",
        "vehicle_first_registration_date", "vehicle_country_first_registration_date",
        "vehicle_last_registration_date", "vehicle_inspection_report_date",
        "vehicle_inspection_expiry_date",
        "vehicle_maker", "vehicle_model", "vehicle_fuel_type",
        "vehicle_primary_color", "vehicle_odometer_verdict_code",
        "coverage", "payment_frequency",
        "province", "municipality",
        "vehicle_maker_model",
    }
    exclude_patterns = ["_price"]
    for other_ins in ALL_INSURERS:
        if other_ins != insurer:
            exclude_exact.add(f"{other_ins}_ded_vs_mean")
            exclude_exact.add(f"{other_ins}_ded_vs_median")
            exclude_exact.add(f"{other_ins}_ded_to_value_ratio")

    feature_cols = []
    for col in df.columns:
        if col in exclude_exact: continue
        if any(pat in col for pat in exclude_patterns): continue
        if df[col].dtype in ["object", "datetime64[ns]"]: continue
        feature_cols.append(col)
    return feature_cols


# =============================================================================
# ENCODING
# =============================================================================
def label_encode_cats(train_df, test_dfs):
    cat_cols = [c for c in train_df.columns if c.endswith("_cat")]
    print(f"  Label-encoding {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        all_vals = set(train_df[col].dropna().unique())
        for tdf in test_dfs:
            if col in tdf.columns:
                all_vals.update(tdf[col].dropna().unique())
        val_to_int = {v: i for i, v in enumerate(sorted(all_vals))}
        val_to_int["nan"] = -1
        val_to_int["__missing__"] = -1
        enc_col = col.replace("_cat", "_encoded")
        train_df[enc_col] = train_df[col].fillna("__missing__").map(val_to_int).fillna(-1).astype(np.int32)
        for tdf in test_dfs:
            if col in tdf.columns:
                tdf[enc_col] = tdf[col].fillna("__missing__").map(val_to_int).fillna(-1).astype(np.int32)


def target_encode(train_df, test_dfs, col, target_col, n_splits=5, smoothing=20):
    new_col = f"{col}_te"
    global_mean = train_df[target_col].mean()
    train_encoded = pd.Series(np.nan, index=train_df.index, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(train_df):
        fold = train_df.iloc[tr_idx]
        stats = fold.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        train_encoded.iloc[va_idx] = train_df.iloc[va_idx][col].map(smoothed)
    train_encoded = train_encoded.fillna(global_mean)
    train_df[new_col] = train_encoded.values

    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    for tdf in test_dfs:
        tdf[new_col] = tdf[col].map(smoothed).fillna(global_mean)
    return new_col


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_lgb_val(X_tr, y_tr, X_va, y_va):
    """Train LGB with early stopping, return model + best iteration."""
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(500)],
    )
    return model, model.best_iteration


def train_lgb_full(X_full, y_full, n_rounds):
    """Train LGB on full data with fixed rounds (no early stopping)."""
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_full, label=y_full, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(500)],
    )
    return model


def train_cb_val(X_tr, y_tr, X_va, y_va):
    """Train CB with early stopping, return model + best iteration."""
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**CB_PARAMS, early_stopping_rounds=CB_EARLY_STOP)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    best_iter = model.get_best_iteration()
    return model, best_iter


def train_cb_full(X_full, y_full, n_rounds):
    """Train CB on full data with fixed rounds (no early stopping)."""
    from catboost import CatBoostRegressor
    params = CB_PARAMS.copy()
    params["iterations"] = n_rounds
    model = CatBoostRegressor(**params)
    model.fit(X_full, y_full)
    return model


def find_best_weight(lgb_preds, cb_preds, y_true):
    best_w, best_mae = 0.5, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        mae = np.mean(np.abs(y_true - (w * lgb_preds + (1 - w) * cb_preds)))
        if mae < best_mae:
            best_mae = mae
            best_w = w
    return best_w, best_mae


# =============================================================================
# CORE: TRAIN ONE MODEL (validation pass + full retrain)
# =============================================================================
def train_one_model(
    quoted_df, price_col, feature_cols, test_b2, test_b3,
    use_log, clip_percentile, label
):
    """
    1. Validation pass (90/10) → find best model, iterations, weight
    2. Full retrain on 100% → predict test blocks
    Returns: (pred_b2, pred_b3, val_scores_dict)
    """
    print(f"\n    --- {label} ({len(quoted_df):,} rows) ---")

    # ---- Outlier clipping ----
    if clip_percentile is not None:
        clip_val = quoted_df[price_col].quantile(clip_percentile / 100.0)
        n_clip = (quoted_df[price_col] > clip_val).sum()
        quoted_df[price_col] = quoted_df[price_col].clip(upper=clip_val)
        print(f"    Clipped {n_clip} rows at {clip_val:.1f}")

    # Remove prices below 5
    too_low = (quoted_df[price_col] < 5) & (quoted_df[price_col].notna())
    n_low = too_low.sum()
    if n_low > 0:
        quoted_df.loc[too_low, price_col] = np.nan
        quoted_df = quoted_df[quoted_df[price_col].notna()].copy()
        print(f"    Removed {n_low} prices below 5")

    # ---- 90/10 validation split ----
    quote_ids = quoted_df["quote_id"]
    threshold = quote_ids.quantile(0.90)
    tr_mask = quote_ids <= threshold
    va_mask = quote_ids > threshold

    X_tr = quoted_df.loc[tr_mask, feature_cols]
    y_tr_raw = quoted_df.loc[tr_mask, price_col]
    X_va = quoted_df.loc[va_mask, feature_cols]
    y_va_raw = quoted_df.loc[va_mask, price_col]

    print(f"    Val pass: train={len(X_tr):,}, val={len(X_va):,}")

    if use_log:
        y_tr = np.log1p(y_tr_raw)
        y_va = np.log1p(y_va_raw)
    else:
        y_tr = y_tr_raw
        y_va = y_va_raw

    # Align test features
    for col in feature_cols:
        if col not in test_b2.columns:
            test_b2[col] = np.nan
        if col not in test_b3.columns:
            test_b3[col] = np.nan

    # ---- VALIDATION PASS: LightGBM ----
    print(f"    [LGB] Validation training...")
    lgb_model_val, lgb_best_iter = train_lgb_val(X_tr, y_tr, X_va, y_va)
    lgb_val_preds = lgb_model_val.predict(X_va)
    if use_log:
        lgb_val_preds = np.expm1(lgb_val_preds)
    lgb_val_preds = np.clip(lgb_val_preds, 0, None)
    lgb_mae = np.mean(np.abs(y_va_raw.values - lgb_val_preds))
    print(f"    [LGB] Val MAE: {lgb_mae:.2f} (best iter: {lgb_best_iter})")

    del lgb_model_val
    gc.collect()

    # ---- VALIDATION PASS: CatBoost ----
    print(f"    [CB] Validation training...")
    cb_model_val, cb_best_iter = train_cb_val(X_tr, y_tr, X_va, y_va)
    cb_val_preds = cb_model_val.predict(X_va)
    if use_log:
        cb_val_preds = np.expm1(cb_val_preds)
    cb_val_preds = np.clip(cb_val_preds, 0, None)
    cb_mae = np.mean(np.abs(y_va_raw.values - cb_val_preds))
    print(f"    [CB] Val MAE: {cb_mae:.2f} (best iter: {cb_best_iter})")

    del cb_model_val
    gc.collect()

    # ---- Find best approach ----
    best_w, ens_mae = find_best_weight(lgb_val_preds, cb_val_preds, y_va_raw.values)

    results = {"lgb": lgb_mae, "cb": cb_mae, "ens": ens_mae}
    best_key = min(results, key=results.get)
    best_mae = results[best_key]

    print(f"    [SELECTION] LGB={lgb_mae:.2f}, CB={cb_mae:.2f}, ENS(w={best_w:.2f})={ens_mae:.2f}")
    print(f"    → WINNER: {best_key} (MAE={best_mae:.2f})")

    val_scores = {
        "lgb_mae": lgb_mae, "cb_mae": cb_mae, "ens_mae": ens_mae,
        "best": best_key, "best_mae": best_mae,
        "lgb_iters": lgb_best_iter, "cb_iters": cb_best_iter,
        "ens_weight_lgb": best_w,
    }

    # ---- FULL RETRAIN on 100% data ----
    X_full = quoted_df[feature_cols]
    y_full_raw = quoted_df[price_col]
    if use_log:
        y_full = np.log1p(y_full_raw)
    else:
        y_full = y_full_raw

    X_test_b2 = test_b2[feature_cols]
    X_test_b3 = test_b3[feature_cols]

    if best_key == "lgb":
        print(f"    [FULL RETRAIN] LGB on {len(X_full):,} rows, {lgb_best_iter} rounds...")
        final_model = train_lgb_full(X_full, y_full, lgb_best_iter)
        pred_b2 = final_model.predict(X_test_b2)
        pred_b3 = final_model.predict(X_test_b3)

        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save_model(os.path.join(MODEL_DIR, f"{label}_lgb.txt"))

    elif best_key == "cb":
        print(f"    [FULL RETRAIN] CB on {len(X_full):,} rows, {cb_best_iter} rounds...")
        final_model = train_cb_full(X_full, y_full, cb_best_iter)
        pred_b2 = final_model.predict(X_test_b2)
        pred_b3 = final_model.predict(X_test_b3)

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save_model(os.path.join(MODEL_DIR, f"{label}_cb.cbm"))

    else:  # ensemble
        print(f"    [FULL RETRAIN] LGB ({lgb_best_iter}r) + CB ({cb_best_iter}r) ensemble (w={best_w:.2f})...")
        lgb_final = train_lgb_full(X_full, y_full, lgb_best_iter)
        cb_final = train_cb_full(X_full, y_full, cb_best_iter)

        lgb_b2 = lgb_final.predict(X_test_b2)
        lgb_b3 = lgb_final.predict(X_test_b3)
        cb_b2 = cb_final.predict(X_test_b2)
        cb_b3 = cb_final.predict(X_test_b3)

        pred_b2 = best_w * lgb_b2 + (1 - best_w) * cb_b2
        pred_b3 = best_w * lgb_b3 + (1 - best_w) * cb_b3

        os.makedirs(MODEL_DIR, exist_ok=True)
        lgb_final.save_model(os.path.join(MODEL_DIR, f"{label}_lgb.txt"))
        cb_final.save_model(os.path.join(MODEL_DIR, f"{label}_cb.cbm"))

    # Reverse log transform
    if use_log:
        pred_b2 = np.expm1(pred_b2)
        pred_b3 = np.expm1(pred_b3)

    pred_b2 = np.clip(pred_b2, 0, None)
    pred_b3 = np.clip(pred_b3, 0, None)

    print(f"    Done. Pred B2 mean={np.mean(pred_b2):.1f}, Pred B3 mean={np.mean(pred_b3):.1f}")

    del final_model
    gc.collect()

    return pred_b2, pred_b3, val_scores


# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()
    print("=" * 70)
    print("  TRAIN FINAL — Production Submission Pipeline")
    print("=" * 70)

    # ---- Load ----
    train, test_b2, test_b3 = load_cleaned_data()
    quote_ids_b2 = test_b2["quote_id"].copy()
    quote_ids_b3 = test_b3["quote_id"].copy()

    # ---- Feature engineering ----
    print("\n[2/8] FEATURE ENGINEERING...")
    t0 = time.time()
    train = engineer_features(train)
    test_b2 = engineer_features(test_b2)
    test_b3 = engineer_features(test_b3)
    print(f"  Done in {time.time()-t0:.0f}s. Train cols: {train.shape[1]}")

    # ---- Label encoding ----
    print("\n[3/8] ENCODING CATEGORICALS...")
    label_encode_cats(train, [test_b2, test_b3])

    # ---- Store coverage info for routing ----
    if "coverage" in test_b2.columns:
        test_b2_coverage = test_b2["coverage"].str.lower().str.strip()
    else:
        test_b2_coverage = test_b2["coverage_cat"] if "coverage_cat" in test_b2.columns else None

    if "coverage" in test_b3.columns:
        test_b3_coverage = test_b3["coverage"].str.lower().str.strip()
    else:
        test_b3_coverage = test_b3["coverage_cat"] if "coverage_cat" in test_b3.columns else None

    # ---- Initialize predictions ----
    preds_b2 = pd.DataFrame({"quote_id": quote_ids_b2})
    preds_b3 = pd.DataFrame({"quote_id": quote_ids_b3})
    all_val_scores = {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==================================================================
    # NORMAL INSURERS (A,B,C,D,F,G,H,J)
    # ==================================================================
    print("\n[4/8] TRAINING NORMAL INSURERS...")

    for insurer in NORMAL_INSURERS:
        price_col = f"{insurer}_price"
        ins_start = time.time()

        print(f"\n{'='*60}")
        print(f"  {insurer} (NORMAL — single model)")
        print(f"{'='*60}")

        if price_col not in train.columns:
            preds_b2[price_col] = 0
            preds_b3[price_col] = 0
            continue

        quoted = train[train[price_col].notna()].copy()
        print(f"  Quoted: {len(quoted):,}")

        # Target encode
        te_cols = []
        for cat_col in ["vehicle_maker_cat", "vehicle_model_cat", "municipality_cat",
                        "province_cat", "vehicle_fuel_type_cat", "postal_code",
                        "vehicle_maker_model_cat"]:
            if cat_col in quoted.columns:
                try:
                    new_col = target_encode(quoted, [test_b2, test_b3], cat_col, price_col)
                    te_cols.append(new_col)
                except:
                    pass

        feature_cols = get_feature_columns(quoted, insurer)
        feature_cols = [c for c in feature_cols if c in quoted.columns]
        print(f"  Features: {len(feature_cols)}")

        # Train (with log transform + clipping for all insurers)
        pred_b2, pred_b3, scores = train_one_model(
            quoted, price_col, feature_cols, test_b2, test_b3,
            use_log=USE_LOG_TRANSFORM,
            clip_percentile=OUTLIER_CLIP_PERCENTILE,
            label=insurer
        )

        preds_b2[price_col] = pred_b2
        preds_b3[price_col] = pred_b3
        all_val_scores[insurer] = scores

        # Cleanup TE columns
        for te_col in te_cols:
            if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
            if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        elapsed = time.time() - ins_start
        print(f"\n  {insurer} total: {elapsed:.0f}s")
        gc.collect()

    # ==================================================================
    # DIFFICULT INSURERS (E,I,K) — COVERAGE SPLIT
    # ==================================================================
    print("\n[5/8] TRAINING DIFFICULT INSURERS (coverage split)...")

    for insurer in DIFFICULT_INSURERS:
        price_col = f"{insurer}_price"
        ins_start = time.time()

        print(f"\n{'='*60}")
        print(f"  {insurer} (DIFFICULT — coverage split)")
        print(f"{'='*60}")

        if price_col not in train.columns:
            preds_b2[price_col] = 0
            preds_b3[price_col] = 0
            continue

        quoted = train[train[price_col].notna()].copy()
        print(f"  Total quoted: {len(quoted):,}")

        # Prepare coverage column
        if "coverage" in quoted.columns:
            quoted_coverage = quoted["coverage"].str.lower().str.strip()
        else:
            quoted_coverage = quoted["coverage_cat"]

        # Initialize per-row predictions
        insurer_pred_b2 = np.zeros(len(test_b2))
        insurer_pred_b3 = np.zeros(len(test_b3))

        insurer_scores = {}
        any_fallback = False

        for cov_type in COVERAGE_TYPES:
            cov_mask_train = quoted_coverage == cov_type
            cov_quoted = quoted[cov_mask_train].copy()

            cov_mask_b2 = test_b2_coverage == cov_type if test_b2_coverage is not None else pd.Series(False, index=test_b2.index)
            cov_mask_b3 = test_b3_coverage == cov_type if test_b3_coverage is not None else pd.Series(False, index=test_b3.index)

            n_cov = len(cov_quoted)
            n_test_b2 = cov_mask_b2.sum()
            n_test_b3 = cov_mask_b3.sum()

            print(f"\n  Coverage: {cov_type} — train={n_cov:,}, test_b2={n_test_b2:,}, test_b3={n_test_b3:,}")

            if n_cov < MIN_ROWS_FOR_COVERAGE_SPLIT:
                print(f"    Too few rows ({n_cov} < {MIN_ROWS_FOR_COVERAGE_SPLIT}), will use combined model")
                any_fallback = True
                continue

            # Target encode within this coverage slice
            te_cols = []
            for cat_col in ["vehicle_maker_cat", "vehicle_model_cat", "municipality_cat",
                            "province_cat", "vehicle_fuel_type_cat", "postal_code",
                            "vehicle_maker_model_cat"]:
                if cat_col in cov_quoted.columns:
                    try:
                        new_col = target_encode(cov_quoted, [test_b2, test_b3], cat_col, price_col)
                        te_cols.append(new_col)
                    except:
                        pass

            feature_cols = get_feature_columns(cov_quoted, insurer)
            feature_cols = [c for c in feature_cols if c in cov_quoted.columns]

            # Train coverage-specific model
            cov_pred_b2, cov_pred_b3, cov_scores = train_one_model(
                cov_quoted, price_col, feature_cols, test_b2, test_b3,
                use_log=USE_LOG_TRANSFORM,
                clip_percentile=OUTLIER_CLIP_PERCENTILE,
                label=f"{insurer}_{cov_type}"
            )

            # Place predictions only for matching coverage rows
            insurer_pred_b2[cov_mask_b2.values] = cov_pred_b2[cov_mask_b2.values]
            insurer_pred_b3[cov_mask_b3.values] = cov_pred_b3[cov_mask_b3.values]

            insurer_scores[cov_type] = cov_scores

            # Cleanup TE
            for te_col in te_cols:
                if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
                if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        # Handle fallback rows (coverage slices too small or unmapped)
        fallback_b2 = insurer_pred_b2 == 0
        fallback_b3 = insurer_pred_b3 == 0

        if fallback_b2.any() or fallback_b3.any() or any_fallback:
            print(f"\n  Training fallback combined model for remaining rows...")
            print(f"    Fallback rows: B2={fallback_b2.sum():,}, B3={fallback_b3.sum():,}")

            # Target encode for fallback
            te_cols = []
            for cat_col in ["vehicle_maker_cat", "vehicle_model_cat", "municipality_cat",
                            "province_cat", "vehicle_fuel_type_cat", "postal_code",
                            "vehicle_maker_model_cat"]:
                if cat_col in quoted.columns:
                    try:
                        new_col = target_encode(quoted, [test_b2, test_b3], cat_col, price_col)
                        te_cols.append(new_col)
                    except:
                        pass

            feature_cols = get_feature_columns(quoted, insurer)
            feature_cols = [c for c in feature_cols if c in quoted.columns]

            fb_pred_b2, fb_pred_b3, fb_scores = train_one_model(
                quoted, price_col, feature_cols, test_b2, test_b3,
                use_log=USE_LOG_TRANSFORM,
                clip_percentile=OUTLIER_CLIP_PERCENTILE,
                label=f"{insurer}_fallback"
            )

            insurer_pred_b2[fallback_b2] = fb_pred_b2[fallback_b2]
            insurer_pred_b3[fallback_b3] = fb_pred_b3[fallback_b3]
            insurer_scores["fallback"] = fb_scores

            for te_col in te_cols:
                if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
                if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        preds_b2[price_col] = np.clip(insurer_pred_b2, 0, None)
        preds_b3[price_col] = np.clip(insurer_pred_b3, 0, None)
        all_val_scores[insurer] = insurer_scores

        elapsed = time.time() - ins_start
        print(f"\n  {insurer} total: {elapsed:.0f}s")
        gc.collect()

    # ==================================================================
    # SAVE SUBMISSIONS
    # ==================================================================
    print("\n[6/8] SAVING SUBMISSIONS...")

    submit_cols = ["quote_id"] + [f"{ins}_price" for ins in ALL_INSURERS]

    for name, preds_df in [("block2", preds_b2), ("block3", preds_b3)]:
        for col in submit_cols:
            if col not in preds_df.columns:
                preds_df[col] = 0

        out = preds_df[submit_cols]
        out_path = os.path.join(OUTPUT_DIR, f"submission_{name}.csv")
        out.to_csv(out_path, sep=";", decimal=".", index=False)
        print(f"  Saved: {out_path} ({len(out):,} rows)")

        check = pd.read_csv(out_path, sep=";", nrows=3)
        print(f"    Verify: {check.columns.tolist()[:4]}... ({len(check.columns)} cols)")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n[7/8] FINAL SUMMARY")
    print("=" * 70)

    for insurer in ALL_INSURERS:
        scores = all_val_scores.get(insurer, {})
        if insurer in DIFFICULT_INSURERS:
            print(f"\n  {insurer} (COVERAGE SPLIT):")
            for key, s in scores.items():
                if isinstance(s, dict):
                    print(f"    {key:15s}: best={s.get('best','?'):>10s}  MAE={s.get('best_mae','?')}")
        else:
            if isinstance(scores, dict) and "best_mae" in scores:
                print(f"  {insurer:15s}: best={scores['best']:>10s}  MAE={scores['best_mae']:.2f}  "
                      f"(LGB={scores.get('lgb_mae','?'):.2f}, CB={scores.get('cb_mae','?'):.2f})")
            else:
                print(f"  {insurer:15s}: {scores}")

    total_time = time.time() - total_start
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print(f"  Models: {MODEL_DIR}/")
    print(f"  Submissions: {OUTPUT_DIR}/")

    # Save scores
    scores_path = os.path.join(OUTPUT_DIR, "validation_scores_final.json")
    with open(scores_path, "w") as f:
        json.dump(all_val_scores, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  DONE! Upload submissions_final/submission_block2.csv to leaderboard!")
    print("=" * 70)


if __name__ == "__main__":
    main()