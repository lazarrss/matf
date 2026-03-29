"""
=============================================================================
TRAIN FINAL v2 — No validation, 100% training, fixed ensemble
=============================================================================
Strategy:
  - Train LGB + CB on 100% of Block 1 (no validation holdout)
  - Fixed ensemble weights per insurer (based on previous experiments)
  - Coverage split for E, I, K
  - Outlier clipping + log transform for all
  - No model selection — always ensemble

Usage:
    1. python clean_data.py   (if not done already)
    2. python train_final_v2.py

Runtime: ~60-90 min on CPU
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

# Fixed ensemble weights (LGB weight — CB weight = 1 - LGB weight)
# Based on previous v3 + test_eik experiments
ENSEMBLE_WEIGHTS = {
    "Insurer_A": 0.35,   # CB was clearly better (6.04 vs 7.00)
    "Insurer_B": 0.40,   # CB generally better
    "Insurer_C": 0.40,
    "Insurer_D": 0.40,
    "Insurer_E": 0.55,   # test_eik: LGB slightly better for E
    "Insurer_F": 0.40,
    "Insurer_G": 0.40,
    "Insurer_H": 0.40,
    "Insurer_I": 0.55,   # test_eik: LGB slightly better for I
    "Insurer_J": 0.40,
    "Insurer_K": 0.40,   # test_eik: CB better for K
}

OUTLIER_CLIP_PERCENTILE = 99.5
USE_LOG_TRANSFORM = True
MIN_ROWS_FOR_COVERAGE_SPLIT = 1000

# LightGBM — fixed rounds (no early stopping)
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
LGB_FIXED_ROUNDS = 5000

# CatBoost — fixed rounds (no early stopping)
CB_PARAMS = {
    "loss_function": "MAE",
    "iterations": 4000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.8,
    "allow_writing_files": False,
    "verbose": 500,
    "random_seed": 42,
    "thread_count": -1,
    "task_type": "CPU",
}


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
    print("[1/6] LOADING CLEANED DATA...")
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

    # Categoricals
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
# MODEL TRAINING (no validation, fixed rounds)
# =============================================================================
def train_lgb(X, y, n_rounds):
    """Train LGB on full data, fixed rounds, no early stopping."""
    import lightgbm as lgb
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(500)],
    )
    return model


def train_cb(X, y):
    """Train CB on full data, fixed rounds, no early stopping."""
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(X, y)
    return model


# =============================================================================
# CORE: TRAIN + PREDICT (no validation)
# =============================================================================
def train_and_predict(
    quoted_df, price_col, feature_cols, test_b2, test_b3,
    lgb_weight, use_log, clip_percentile, label
):
    """
    Train LGB + CB on 100% data, ensemble with fixed weight, predict test.
    Returns: (pred_b2, pred_b3)
    """
    print(f"\n    --- {label} ({len(quoted_df):,} rows, LGB_w={lgb_weight:.2f}) ---")

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

    # ---- Prepare data ----
    X = quoted_df[feature_cols]
    y_raw = quoted_df[price_col]

    if use_log:
        y = np.log1p(y_raw)
    else:
        y = y_raw

    print(f"    Training on {len(X):,} rows, {len(feature_cols)} features")

    # Align test features
    for col in feature_cols:
        if col not in test_b2.columns:
            test_b2[col] = np.nan
        if col not in test_b3.columns:
            test_b3[col] = np.nan

    X_test_b2 = test_b2[feature_cols]
    X_test_b3 = test_b3[feature_cols]

    # ---- Train LightGBM ----
    print(f"    [LGB] Training {LGB_FIXED_ROUNDS} rounds...")
    lgb_model = train_lgb(X, y, LGB_FIXED_ROUNDS)

    lgb_pred_b2 = lgb_model.predict(X_test_b2)
    lgb_pred_b3 = lgb_model.predict(X_test_b3)

    if use_log:
        lgb_pred_b2 = np.expm1(lgb_pred_b2)
        lgb_pred_b3 = np.expm1(lgb_pred_b3)
    lgb_pred_b2 = np.clip(lgb_pred_b2, 0, None)
    lgb_pred_b3 = np.clip(lgb_pred_b3, 0, None)

    # Save LGB
    os.makedirs(MODEL_DIR, exist_ok=True)
    lgb_model.save_model(os.path.join(MODEL_DIR, f"{label}_lgb.txt"))
    print(f"    [LGB] Done. Pred B2 mean={np.mean(lgb_pred_b2):.1f}")

    del lgb_model
    gc.collect()

    # ---- Train CatBoost ----
    print(f"    [CB] Training {CB_PARAMS['iterations']} rounds...")
    cb_model = train_cb(X, y)

    cb_pred_b2 = cb_model.predict(X_test_b2)
    cb_pred_b3 = cb_model.predict(X_test_b3)

    if use_log:
        cb_pred_b2 = np.expm1(cb_pred_b2)
        cb_pred_b3 = np.expm1(cb_pred_b3)
    cb_pred_b2 = np.clip(cb_pred_b2, 0, None)
    cb_pred_b3 = np.clip(cb_pred_b3, 0, None)

    # Save CB
    cb_model.save_model(os.path.join(MODEL_DIR, f"{label}_cb.cbm"))
    print(f"    [CB] Done. Pred B2 mean={np.mean(cb_pred_b2):.1f}")

    del cb_model
    gc.collect()

    # ---- Ensemble ----
    cb_weight = 1.0 - lgb_weight
    pred_b2 = lgb_weight * lgb_pred_b2 + cb_weight * cb_pred_b2
    pred_b3 = lgb_weight * lgb_pred_b3 + cb_weight * cb_pred_b3

    pred_b2 = np.clip(pred_b2, 0, None)
    pred_b3 = np.clip(pred_b3, 0, None)

    print(f"    [ENS] w_lgb={lgb_weight:.2f}, w_cb={cb_weight:.2f}")
    print(f"    [ENS] Pred B2 mean={np.mean(pred_b2):.1f}, Pred B3 mean={np.mean(pred_b3):.1f}")

    return pred_b2, pred_b3


# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()
    print("=" * 70)
    print("  TRAIN FINAL v2 — 100% Training, Fixed Ensemble")
    print("=" * 70)

    # ---- Load ----
    train, test_b2, test_b3 = load_cleaned_data()
    quote_ids_b2 = test_b2["quote_id"].copy()
    quote_ids_b3 = test_b3["quote_id"].copy()

    # ---- Feature engineering ----
    print("\n[2/6] FEATURE ENGINEERING...")
    t0 = time.time()
    train = engineer_features(train)
    test_b2 = engineer_features(test_b2)
    test_b3 = engineer_features(test_b3)
    print(f"  Done in {time.time()-t0:.0f}s. Train cols: {train.shape[1]}")

    # ---- Label encoding ----
    print("\n[3/6] ENCODING CATEGORICALS...")
    label_encode_cats(train, [test_b2, test_b3])

    # ---- Coverage info for routing ----
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==================================================================
    # NORMAL INSURERS
    # ==================================================================
    print("\n[4/6] TRAINING NORMAL INSURERS...")

    for insurer in NORMAL_INSURERS:
        price_col = f"{insurer}_price"
        ins_start = time.time()

        print(f"\n{'='*60}")
        print(f"  {insurer} (NORMAL)")
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

        lgb_w = ENSEMBLE_WEIGHTS[insurer]

        pred_b2, pred_b3 = train_and_predict(
            quoted, price_col, feature_cols, test_b2, test_b3,
            lgb_weight=lgb_w,
            use_log=USE_LOG_TRANSFORM,
            clip_percentile=OUTLIER_CLIP_PERCENTILE,
            label=insurer
        )

        preds_b2[price_col] = pred_b2
        preds_b3[price_col] = pred_b3

        # Cleanup TE
        for te_col in te_cols:
            if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
            if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        elapsed = time.time() - ins_start
        print(f"\n  {insurer} done in {elapsed:.0f}s")
        gc.collect()

    # ==================================================================
    # DIFFICULT INSURERS — COVERAGE SPLIT
    # ==================================================================
    print("\n[5/6] TRAINING DIFFICULT INSURERS (coverage split)...")

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

        if "coverage" in quoted.columns:
            quoted_coverage = quoted["coverage"].str.lower().str.strip()
        else:
            quoted_coverage = quoted["coverage_cat"]

        insurer_pred_b2 = np.zeros(len(test_b2))
        insurer_pred_b3 = np.zeros(len(test_b3))
        any_fallback = False

        lgb_w = ENSEMBLE_WEIGHTS[insurer]

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
                print(f"    Too few rows, will use fallback")
                any_fallback = True
                continue

            # Target encode within coverage slice
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

            cov_pred_b2, cov_pred_b3 = train_and_predict(
                cov_quoted, price_col, feature_cols, test_b2, test_b3,
                lgb_weight=lgb_w,
                use_log=USE_LOG_TRANSFORM,
                clip_percentile=OUTLIER_CLIP_PERCENTILE,
                label=f"{insurer}_{cov_type}"
            )

            insurer_pred_b2[cov_mask_b2.values] = cov_pred_b2[cov_mask_b2.values]
            insurer_pred_b3[cov_mask_b3.values] = cov_pred_b3[cov_mask_b3.values]

            # Cleanup TE
            for te_col in te_cols:
                if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
                if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        # Fallback for unmapped rows
        fallback_b2 = insurer_pred_b2 == 0
        fallback_b3 = insurer_pred_b3 == 0

        if fallback_b2.any() or fallback_b3.any() or any_fallback:
            print(f"\n  Fallback combined model...")
            print(f"    Fallback rows: B2={fallback_b2.sum():,}, B3={fallback_b3.sum():,}")

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

            fb_pred_b2, fb_pred_b3 = train_and_predict(
                quoted, price_col, feature_cols, test_b2, test_b3,
                lgb_weight=lgb_w,
                use_log=USE_LOG_TRANSFORM,
                clip_percentile=OUTLIER_CLIP_PERCENTILE,
                label=f"{insurer}_fallback"
            )

            insurer_pred_b2[fallback_b2] = fb_pred_b2[fallback_b2]
            insurer_pred_b3[fallback_b3] = fb_pred_b3[fallback_b3]

            for te_col in te_cols:
                if te_col in test_b2.columns: test_b2.drop(columns=[te_col], inplace=True)
                if te_col in test_b3.columns: test_b3.drop(columns=[te_col], inplace=True)

        preds_b2[price_col] = np.clip(insurer_pred_b2, 0, None)
        preds_b3[price_col] = np.clip(insurer_pred_b3, 0, None)

        elapsed = time.time() - ins_start
        print(f"\n  {insurer} done in {elapsed:.0f}s")
        gc.collect()

    # ==================================================================
    # SAVE SUBMISSIONS
    # ==================================================================
    print("\n[6/6] SAVING SUBMISSIONS...")

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
        print(f"    Sample: quote_id={check['quote_id'].iloc[0]}, "
              f"A={check['Insurer_A_price'].iloc[0]:.2f}, "
              f"E={check['Insurer_E_price'].iloc[0]:.2f}")

    total_time = time.time() - total_start
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print(f"  Models saved to '{MODEL_DIR}/'")
    print(f"  Submissions in '{OUTPUT_DIR}/'")
    print(f"\n  Ensemble weights used:")
    for ins, w in ENSEMBLE_WEIGHTS.items():
        print(f"    {ins}: LGB={w:.2f}, CB={1-w:.2f}")

    print("\n" + "=" * 70)
    print("  DONE! Upload submissions_final/submission_block2.csv to leaderboard!")
    print("=" * 70)


if __name__ == "__main__":
    main()