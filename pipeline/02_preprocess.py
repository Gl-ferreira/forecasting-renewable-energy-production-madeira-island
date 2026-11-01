"""
02_preprocess.py

Step 2: cleaning, train/validation/test split, cyclical encoding and
imputation (manuscript Section 2.3).

- Casts numeric columns to float32 and drops variables not used in the
  study (Month Name, Hour of the Year, Year, Curtailed Wind Energy,
  snowdepth, Hour of the day).
- Splits the hourly data chronologically into 70% training, 15% validation
  and 15% test, at the same hourly boundaries as the original analysis
  (2023-07-27 06:00 and 2024-02-13 02:00). In daily terms this is the split
  reported in Section 2.3.
- Encodes Season as an integer and replaces Month, Season, day of month
  and day of week with sine/cosine pairs; a weekend flag is also added.
  The day-of-month, day-of-week and weekend features were part of the
  original candidate set but were not selected for any final model
  (manuscript Table 1).
- Imputes, using the training split only: linear interpolation for
  visibility, solarradiation, solarenergy and uvindex, and Random Forest
  imputation for Hydropower (from temp, dew, Season_cos, Month_sin and
  Month_cos, trained without November 2021) and for windgust (from
  windspeed and Wind). As in the original analysis, validation and test are
  not imputed at this stage, since both gaps fall inside the training
  period; the script prints any values still missing in those splits.

Input:  data/merged_hourly.csv
Output: data/train_hourly.csv, data/val_hourly.csv, data/test_hourly.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "merged_hourly.csv"

TRAIN_END = "2023-07-27 06:00"
VAL_START = "2023-07-27 07:00"
VAL_END = "2024-02-13 02:00"
TEST_START = "2024-02-13 03:00"

COLUMNS_TO_DROP = [
    "Month Name",
    "Hour of the Year",
    "Year",
    "Curtailed Wind Energy",
    "snowdepth",
    "Hour of the day",
]

SEASON_ENCODING = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}

INTERPOLATE_COLS = ["visibility", "solarradiation", "solarenergy", "uvindex"]

HYDROPOWER_RF_FEATURES = ["temp", "dew", "Season_cos", "Month_sin", "Month_cos"]
WINDGUST_RF_FEATURES = ["windspeed", "Wind"]


def load_merged() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH, index_col=0, parse_dates=True)
    return df


def initial_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2.1 - datatypes + dropping unused columns."""
    df = df.copy()
    exclude = ["Season", "Month Name"]
    float_cols = df.select_dtypes(include=["float64"]).columns.difference(exclude)
    df[float_cols] = df[float_cols].astype("float32")

    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            del df[col]
    return df


def train_val_test_split(df: pd.DataFrame):
    """Notebook 2.2 - fixed-date 70/15/15 split."""
    train = df.loc[:TRAIN_END].copy()
    val = df.loc[VAL_START:VAL_END].copy()
    test = df.loc[TEST_START:].copy()

    for part in (train, val, test):
        for col in ("Day of Month", "Month"):
            part[col] = part[col].astype(int)

    return train, val, test


def encode_cyclical_feature(parts, col, max_val):
    for df in parts:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val).astype("float32")
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val).astype("float32")
        df.drop(columns=[col], inplace=True)


def data_transformation(train, val, test):
    """Notebook 2.2 (duplicate heading) - Season encoding, day_of_week/is_weekend,
    cyclical sin/cos encoding."""
    parts = [train, val, test]

    for df in parts:
        df["Season"] = df["Season"].map(SEASON_ENCODING).astype("int32")

    for df in parts:
        df["day_of_week"] = df.index.dayofweek

    for df in parts:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int32")

    encode_cyclical_feature(parts, "Day of Month", 31)
    encode_cyclical_feature(parts, "Month", 12)
    encode_cyclical_feature(parts, "Season", 4)
    encode_cyclical_feature(parts, "day_of_week", 7)

    return train, val, test


def linear_interpolate_train(train: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2.3.1."""
    train = train.copy()
    train[INTERPOLATE_COLS] = train[INTERPOLATE_COLS].interpolate(
        method="linear", limit_direction="both"
    )
    return train


def rf_impute_hydropower(train: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2.3.2 - Hydropower RF imputer, trained excluding Nov 2021,
    applied to Nov 2021 rows that are NaN or exactly 0."""
    train = train.copy()
    target = "Hydropower"
    features = HYDROPOWER_RF_FEATURES

    df_train = train[train[target].notna()].copy()
    mask_nov_2021 = (df_train.index.month == 11) & (df_train.index.year == 2021)
    X_train = df_train.loc[~mask_nov_2021, features]
    y_train = df_train.loc[~mask_nov_2021, target]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    mask_missing_nov2021 = (
        (train.index.month == 11)
        & (train.index.year == 2021)
        & (train[target].isna() | train[target].eq(0))
    )
    X_missing = train.loc[mask_missing_nov2021, features]

    n_imputed = 0
    if not X_missing.empty:
        y_pred = model.predict(X_missing)
        train.loc[mask_missing_nov2021, target] = y_pred.astype(np.float32)
        n_imputed = int(mask_missing_nov2021.sum())

    return train, n_imputed


def rf_impute_windgust(train: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2.3.2 - windgust RF imputer, trained on all non-null rows,
    applied to all remaining NaN rows."""
    train = train.copy()
    features = WINDGUST_RF_FEATURES

    train_known = train[train["windgust"].notna()]
    X_train = train_known[features]
    y_train = train_known["windgust"]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    mask_missing = train["windgust"].isna()
    n_imputed = int(mask_missing.sum())
    if n_imputed:
        X_missing = train.loc[mask_missing, features]
        y_missing_pred = rf_model.predict(X_missing)
        train.loc[mask_missing, "windgust"] = y_missing_pred.astype(np.float32)

    return train, n_imputed


def main():
    stats = {}
    df = load_merged()
    stats["merged_rows"] = len(df)

    df = initial_adjustments(df)
    stats["after_initial_adjustments_cols"] = df.shape[1]

    train, val, test = train_val_test_split(df)
    stats["train_rows"] = len(train)
    stats["val_rows"] = len(val)
    stats["test_rows"] = len(test)

    train, val, test = data_transformation(train, val, test)
    stats["cols_after_cyclical_encoding"] = train.shape[1]

    train = linear_interpolate_train(train)
    stats["train_na_after_interpolation"] = {
        c: int(train[c].isna().sum()) for c in INTERPOLATE_COLS
    }

    train, n_hydro_imputed = rf_impute_hydropower(train)
    stats["hydropower_rf_imputed_rows"] = n_hydro_imputed
    stats["hydropower_remaining_na_train"] = int(train["Hydropower"].isna().sum())

    train, n_windgust_imputed = rf_impute_windgust(train)
    stats["windgust_rf_imputed_rows"] = n_windgust_imputed
    stats["windgust_remaining_na_train"] = int(train["windgust"].isna().sum())

    # Diagnostics: leftover NaNs in val/test (not imputed at this stage,
    # matching the original notebook's scope - see module docstring)
    stats["val_remaining_na_total"] = int(val.isna().sum().sum())
    stats["test_remaining_na_total"] = int(test.isna().sum().sum())
    stats["val_na_by_col"] = {c: int(v) for c, v in val.isna().sum().items() if v > 0}
    stats["test_na_by_col"] = {c: int(v) for c, v in test.isna().sum().items() if v > 0}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_DIR / "train_hourly.csv")
    val.to_csv(DATA_DIR / "val_hourly.csv")
    test.to_csv(DATA_DIR / "test_hourly.csv")

    print("=== 02_preprocess.py ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return train, val, test, stats


if __name__ == "__main__":
    main()
