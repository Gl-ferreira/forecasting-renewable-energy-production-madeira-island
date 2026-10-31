"""
03_feature_engineering.py
==========================
Reproduces notebook sections 2.4 (daily aggregation) and 2.5 (feature
engineering - lags and rolling means), cells ~129-137.

2.4 - Daily aggregation (cells 129-131)
    The hourly train/val/test frames are each resampled to daily frequency
    with a per-column aggregator (`variables_to_agg` in the original):
      - production targets (Hydropower, Fossil, Natural Gas, Biomass, Wind,
        Solar) and precip: 'sum'
      - continuous weather variables (temp, feelslike, humidity, dew,
        windgust, windspeed, pressure, visibility, cloudcover,
        solarradiation, solarenergy, winddir_sin, winddir_cos): 'mean'
      - capacity columns, precipprob, uvindex, is_weekend, and all the
        already-cyclical-encoded calendar columns (Day of Month_sin/cos,
        Month_sin/cos, Season_sin/cos, day_of_week_sin/cos): 'first'
    Reproduced verbatim from the notebook's `variables_to_agg` dict
    (including its duplicate 'dew' and 'is_weekend' keys, which in a Python
    dict simply keep the last-assigned aggregator - harmless here since
    both duplicate assignments use the same aggregator each time).

2.5 - Feature engineering (cells 135-137)
    Lag (1, 3, 7 days) and rolling-mean (3, 7 days, using the PRIOR day's
    rolling window via `shift(1)` so the current day's own value never
    leaks into its own rolling mean) features are created for:
      Hydropower, Wind, Solar, Biomass, humidity, solarradiation, windgust,
      windspeed
    Lags/rolling means are computed on train+val+test concatenated
    together (so that val's and test's first few lagged values can be
    filled from the end of the preceding split), then split back apart.
    The first 7 rows of `train` are then dropped (they are now NaN for the
    lag-7 / roll-7 features, since train has nothing further back to pull
    from).

Input:  data/train_hourly.csv, data/val_hourly.csv, data/test_hourly.csv
Output: data/train_daily.csv, data/val_daily.csv, data/test_daily.csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VARIABLES_TO_AGG = {
    "Hydropower": "sum",
    "Fossil": "sum",
    "Natural Gas": "sum",
    "Biomass": "sum",
    "Wind": "sum",
    "Solar": "sum",
    "Wind Energy Capacity": "first",
    "Hydropower Energy Capacity": "first",
    "Solar Energy Capacity": "first",
    "Biomass Energy Capacity": "first",
    "Fossil Energy Capacity": "first",
    "Natural Gas Energy Capacity": "first",
    "temp": "mean",
    "feelslike": "mean",
    "humidity": "mean",
    "dew": "mean",
    "precip": "sum",
    "precipprob": "first",
    "windgust": "mean",
    "windspeed": "mean",
    "pressure": "mean",
    "visibility": "mean",
    "cloudcover": "mean",
    "solarradiation": "mean",
    "solarenergy": "mean",
    "uvindex": "first",
    "is_weekend": "first",
    "winddir_sin": "mean",
    "winddir_cos": "mean",
    "Day of Month_sin": "first",
    "Day of Month_cos": "first",
    "Month_sin": "first",
    "Month_cos": "first",
    "Season_sin": "first",
    "Season_cos": "first",
    "day_of_week_sin": "first",
    "day_of_week_cos": "first",
}

LAG_ROLL_COLS = [
    "Hydropower",
    "Wind",
    "Solar",
    "Biomass",
    "humidity",
    "solarradiation",
    "windgust",
    "windspeed",
]
LAGS = [1, 3, 7]
ROLLS = [3, 7]


def load_hourly_splits():
    train = pd.read_csv(DATA_DIR / "train_hourly.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_DIR / "val_hourly.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_DIR / "test_hourly.csv", index_col=0, parse_dates=True)
    return train, val, test


def daily_aggregate(train, val, test):
    """Notebook 2.4."""
    cols = [c for c in VARIABLES_TO_AGG if c in train.columns]
    agg = {c: VARIABLES_TO_AGG[c] for c in cols}
    train_d = train.resample("D").agg(agg)
    val_d = val.resample("D").agg(agg)
    test_d = test.resample("D").agg(agg)
    return train_d, val_d, test_d


def create_lag_rolling_features(train, val, test, cols, lags=LAGS, rolls=ROLLS, full_window=True):
    """Notebook 2.5 - verbatim port of `create_lag_rolling_features`."""
    train = train.sort_index()
    val = val.sort_index()
    test = test.sort_index()

    combined = pd.concat([train, val, test], axis=0)
    for col in cols:
        for lag in lags:
            combined[f"{col}_lag{lag}"] = combined[col].shift(lag)
        for w in rolls:
            minp = w if full_window else 1
            combined[f"{col}_roll{w}"] = (
                combined[col].shift(1).rolling(window=w, min_periods=minp).mean()
            )

    train_out = combined.loc[train.index]
    val_out = combined.loc[val.index]
    test_out = combined.loc[test.index]
    return train_out, val_out, test_out


def main():
    stats = {}
    train, val, test = load_hourly_splits()
    stats["train_hourly_rows"] = len(train)
    stats["val_hourly_rows"] = len(val)
    stats["test_hourly_rows"] = len(test)

    train_d, val_d, test_d = daily_aggregate(train, val, test)
    stats["train_daily_rows"] = len(train_d)
    stats["val_daily_rows"] = len(val_d)
    stats["test_daily_rows"] = len(test_d)
    stats["daily_cols"] = train_d.shape[1]

    train_d, val_d, test_d = create_lag_rolling_features(train_d, val_d, test_d, LAG_ROLL_COLS)
    stats["cols_after_lag_roll"] = train_d.shape[1]

    rows_before_trim = len(train_d)
    train_d = train_d.iloc[7:]
    stats["train_rows_dropped_for_lag7_nans"] = rows_before_trim - len(train_d)
    stats["train_daily_rows_final"] = len(train_d)
    stats["val_daily_rows_final"] = len(val_d)
    stats["test_daily_rows_final"] = len(test_d)

    # Diagnostic: boundary calendar dates that land in two adjacent splits
    # at once. Each split is resampled to daily frequency independently
    # (see daily_aggregate above), so the single calendar day on which
    # train ends and val begins (and likewise val/test) gets a *partial*
    # day in BOTH neighbouring splits, and after the lag/rolling merge
    # both partial-day rows share that date label. This is inherited
    # as-is from the original notebook's own `create_lag_rolling_features`
    # + resample('D') combination - not something introduced here - but
    # is flagged because any later step that does an index-based
    # join/merge on these frames (rather than a positional one) must
    # account for it; see 04_scale_and_select.py's `_attach_target` note.
    stats["train_daily_duplicate_dates"] = [
        str(d) for d in train_d.index[train_d.index.duplicated(keep=False)].unique()
    ]
    stats["val_daily_duplicate_dates"] = [
        str(d) for d in val_d.index[val_d.index.duplicated(keep=False)].unique()
    ]
    stats["test_daily_duplicate_dates"] = [
        str(d) for d in test_d.index[test_d.index.duplicated(keep=False)].unique()
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_d.to_csv(DATA_DIR / "train_daily.csv")
    val_d.to_csv(DATA_DIR / "val_daily.csv")
    test_d.to_csv(DATA_DIR / "test_daily.csv")

    print("=== 03_feature_engineering.py ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return train_d, val_d, test_d, stats


if __name__ == "__main__":
    main()
