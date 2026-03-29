import pandas as pd

parquet_files = ["block1_train.parquet", "block2_test.parquet", "block3_test.parquet"]

for file in parquet_files:
    # Read parquet
    df = pd.read_parquet(file)

    # Convert all columns to string
    df = df.astype(str)

    # Optional: replace "nan" with empty string
    df = df.replace("nan", "")

    # Save as CSV (same name)
    output_file = file.replace(".parquet", ".csv")
    df.to_csv(output_file, index=False)

    print(f"Converted {file} -> {output_file}")