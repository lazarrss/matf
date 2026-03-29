"""
=============================================================================
CLEAN DATA — Drop useless columns, save cleaned copies
=============================================================================
Usage: python clean_data.py
=============================================================================
"""

import pandas as pd
import os
import sys
import time

DATA_DIR = "data"
CLEAN_DIR = "data_cleaned"

DROP_COLS = [
    "vehicle_number_plate",
    "vehicle_number_of_wheels",
    "vehicle_is_taxi",
    "vehicle_net_max_power_electric",
    "vehicle_nominal_continuous_max_power",
    "postal_code_houses_owned_by_rental_association_ratio",
    "postal_code_houses_built_before_1945_ratio",
    "postal_code_houses_built_45_to_65_ratio",
    "postal_code_houses_built_65_to_75_ratio",
    "postal_code_houses_built_75_to_85_ratio",
    "postal_code_houses_built_85_to_95_ratio",
    "postal_code_houses_built_95_to_05_ratio",
    "postal_code_houses_built_05_to_15_ratio",
    "has_second_driver",
    "second_driver_age",
    "second_driver_claim_free_years",
    "second_driver_birthdate",
    "usage",
    "vehicle_gross_weight",
]


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


def main():
    t0 = time.time()
    print("=" * 60)
    print("  CLEAN DATA — Drop useless columns")
    print("=" * 60)

    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [
        ("block1_train.csv", "block1_train_cleaned.csv"),
        ("block2_test.csv", "block2_test_cleaned.csv"),
        ("block3_test.csv", "block3_test_cleaned.csv"),
    ]

    for in_name, out_name in files:
        in_path = os.path.join(DATA_DIR, in_name)
        out_path = os.path.join(CLEAN_DIR, out_name)

        if not os.path.exists(in_path):
            print(f"  ERROR: {in_path} not found!")
            sys.exit(1)

        sep = detect_separator(in_path)
        print(f"\n  Loading {in_name}...")
        df = pd.read_csv(in_path, sep=sep, low_memory=False)
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        print(f"    Before: {df.shape[0]:,} rows × {df.shape[1]} cols")

        dropped = 0
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                dropped += 1

        print(f"    Dropped {dropped} columns")
        print(f"    After:  {df.shape[0]:,} rows × {df.shape[1]} cols")

        df.to_csv(out_path, index=False)
        print(f"    Saved → {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Cleaned files in '{CLEAN_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()