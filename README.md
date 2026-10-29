# Insurance Market Reverse Engineering — Hackathon Solution

Our team's solution for a two-day data science hackathon (28–29 March 2026): **reverse-engineering the car insurance pricing of 11 real market competitors** using only their quoted premiums and the customer, vehicle and regional data behind each quote.

## The Challenge

- **Data:** real quotes from an insurance comparison platform. Each row is one quote request (driver, vehicle, location and area-level statistics), with the price and deductible offered by each insurer (*Insurer A–K*). Missing prices mean the insurer declined to quote.
- **Task 1, predictive modeling:** predict the premium each insurer would quote for unseen profiles. Evaluated by **Mean Absolute Error (MAE)** on a public (week 5) and private (week 6) leaderboard.
- **Task 2, strategic insight:** identify which competitors over- or under-price which customer segments and recommend a pricing strategy for a new insurer.

## Our Approach

- **Data cleaning** (`preprocess.py`): dropped uninformative and identifier columns and saved cleaned copies of the three data blocks.
- **Feature engineering:** driver age buckets and young/senior driver flags, vehicle ratios (power-to-weight, power per cc, value per year of age), inspection validity, log transforms of skewed features (vehicle value, power, mileage), label encoding and **K-fold target encoding** of high-cardinality categorical features.
- **One model per insurer**, trained only on rows where that insurer actually quoted.
- **Gradient boosting ensemble:** **LightGBM** and **CatBoost** trained directly on MAE, with a validation pass to choose the winning model or the best blend weight and number of iterations, then retraining on 100% of the data.
- **Special handling of difficult insurers (E, I, K):** separate models per coverage type (MTPL, limited casco, casco), outlier clipping at the 99.5th percentile, and a log-transformed target.

## Files

| File | Description |
|---|---|
| `preprocess.py` | Data cleaning |
| `solution.py` | Main pipeline: validation-based model selection, retraining and submission generation |
| `solution_shorter.py` | Faster variant: fixed LightGBM + CatBoost ensemble on 100% of the data, without a validation pass |
| `parquet_to_csv.py` | Converts the provided Parquet data blocks to CSV |
| `preview_outputs/` | Preview of the raw data structure |
| `Hackathon Final Task.pdf` | Official task description |

## Running

The competition data is not included in the repository. Place `block1_train.csv`, `block2_test.csv` and `block3_test.csv` in a `data/` folder (use `parquet_to_csv.py` to convert the original Parquet files).

```bash
pip install -r requirements.txt
python preprocess.py   # writes cleaned data to data_cleaned/
python solution.py     # trains models and writes predictions to submissions_final/
```

Full training takes about 1.5–2 hours on a CPU.

## Tech Stack

Python · pandas · NumPy · scikit-learn · LightGBM · CatBoost · matplotlib · Plotly
