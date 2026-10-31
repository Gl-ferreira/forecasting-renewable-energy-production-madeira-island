"""
03_feature_engineering.py

Step 3: turns raw hourly readings into the lagged, rolling-window features
that actually let a model learn what usually happens next for a given
energy source, instead of just reacting to the latest timestamp. Reproduces
notebook sections 2.4 (daily aggregation) and 2.5 (lags and rolling means),
roughly cells 129-137.

Daily aggregation (2.4) resamples each hourly train/val/test frame to daily
frequency with a per-column aggregator, matching the notebook's
`variables_to_agg` dict column for column: the production targets
(Hydropower, Fossil, Natural Gas, Biomass, Wind, Solar) and precip sum over
the day, the continuous weather variables (temp, feelslike, humidity, dew,
windgust, windspeed, pressure, visibility, cloudcover, solarradiation,
solarenergy, winddir_sin, winddir_cos) average, and the capacity columns,
precipprob, uvindex, is_weekend, and the already-cyclical-encoded calendar
columns (Day of Month_sin/cos, Month_sin/cos, Season_sin/cos,
day_of_week_sin/cos) just take the first value of the day. The dict is
copied verbatim, including its duplicate 'dew' and 'is_weekend' keys; a
Python dict just keeps whichever assignment came last, which is harmless
here since both duplicates use the same aggregator anyway.

Feature engineering (2.5) adds 1/3/7-day lags and 3/7-day rolling means for
Hydropower, Wind, Solar, Biomass, humidity, solarradiation, windgust and
windspeed. The rolling means are taken over the prior day's window
(`shift(1)` first) so a day's own value never leaks into its own rolling
mean. Lags and rolling means are computed on train+val+test concatenated
together, so val's and test's first few lagged values can be filled in from
the tail of the split before them, then the three frames are split back
apart. The first 7 rows of train are dropped afterward, since they're left
with NaNs in the lag-7/roll-7 columns once train has nothing further back to
draw on.

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

    # Worth a flag: the boundary calendar date between two adjacent splits
    # shows up twice. Each split gets resampled to daily frequency on its
    # own (see daily_aggregate above), so the single day where train ends
    # and val begins, or val ends and test begins, has a partial day's
    # worth of hours on each side, and both partial-day rows end up
    # sharing the same date label after the lag/rolling merge. That's
    # inherited straight from the notebook's own combination of
    # create_lag_rolling_features and resample('D'), not something this
    # script introduces, but anything downstream that joins or merges on
    # these frames by index rather than by position needs to know about
    # it - see the note on _attach_target in 04_scale_and_select.py.
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
