"""
01_merge_data.py

Step 1: merges the two weather stations and the production data on a
common hourly timestamp (manuscript Sections 2.1, 2.2 and the first
paragraph of 2.3).

1. Loads the two hourly weather files and the 15-minute production file.
2. Converts winddir to sine and cosine for each station, since a circular
   variable cannot be averaged in degrees, drops the original columns and
   averages the two stations. As in the original analysis, the averaged
   sine/cosine pair is not renormalized.
3. Removes duplicated timestamps (first record kept). The EEM values are
   average power over each 15-minute interval (MW), so they are multiplied
   by 0.25 h to obtain energy (MWh) before being summed into hourly bins.
   Hours with no 15-minute records are dropped. An hour with only missing
   values sums to 0 rather than NaN, which affects Hydropower in November
   2021; 02_preprocess.py treats those zeros as missing before imputation.
4. Keeps only the timestamps present in both datasets and joins them.

Difference from the original analysis: the real data had a single DST
mismatch between production and weather (March 2023), which was removed by
date. The synthetic data have a mismatch every spring, so this script
removes any hour that exists in only one of the two datasets around each
DST change. This also removes one trailing hour on the last day, caused by
the two synthetic generators ending at slightly different times.

Output: data/merged_hourly.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRODUCTION_CSV = DATA_DIR / "synthetic_production.csv"
WEATHER_FW_CSV = DATA_DIR / "synthetic_weather_FW_Funchal.csv"
WEATHER_LPMA_CSV = DATA_DIR / "synthetic_weather_LPMA.csv"
OUT_PATH = DATA_DIR / "merged_hourly.csv"

PRODUCTION_FLOW_COLS = ["Hydropower", "Fossil", "Natural Gas", "Biomass", "Wind", "Solar"]

PRODUCTION_HOURLY_AGG = {
    "Date": "first",
    "Hydropower": "sum",
    "Fossil": "sum",
    "Natural Gas": "sum",
    "Biomass": "sum",
    "Wind": "sum",
    "Solar": "sum",
    "Curtailed Wind Energy": "sum",
    "Year": "first",
    "Wind Energy Capacity": "first",
    "Hydropower Energy Capacity": "first",
    "Solar Energy Capacity": "first",
    "Biomass Energy Capacity": "first",
    "Fossil Energy Capacity": "first",
    "Natural Gas Energy Capacity": "first",
    "Month Name": "first",
    "Day of Month": "first",
    "Month": "first",
    "Season": "first",
    "Hour of the day": "first",
    "Hour of the Year": "first",
}


def load_raw():
    production = pd.read_csv(PRODUCTION_CSV, parse_dates=["Hora"]).set_index("Hora")
    fw_funchal = pd.read_csv(WEATHER_FW_CSV, parse_dates=["date_time"]).set_index("date_time")
    lpma = pd.read_csv(WEATHER_LPMA_CSV, parse_dates=["date_time"]).set_index("date_time")
    return production, fw_funchal, lpma


def merge_weather(fw_funchal: pd.DataFrame, lpma: pd.DataFrame) -> pd.DataFrame:
    """Notebook 1.2.1 - circular handling of winddir, then plain mean of stations."""
    fw_funchal = fw_funchal.copy()
    lpma = lpma.copy()

    for station in (fw_funchal, lpma):
        station["winddir_rad"] = np.deg2rad(station["winddir"])
        station["winddir_sin"] = np.sin(station["winddir_rad"])
        station["winddir_cos"] = np.cos(station["winddir_rad"])
        station.drop(columns=["winddir", "winddir_rad"], inplace=True)

    weather_data = (fw_funchal + lpma) / 2
    return weather_data


def merge_production_weather(production: pd.DataFrame, weather_data: pd.DataFrame):
    """Notebook 1.2.2 - dedup, rate->mean, hourly resample, DST alignment, concat."""
    stats = {}

    stats["production_raw_rows"] = len(production)
    stats["weather_raw_rows"] = len(weather_data)

    # (a) duplicate handling - keep first, both datasets
    n_dup_production = int(production.index.duplicated().sum())
    n_dup_weather = int(weather_data.index.duplicated().sum())
    production = production[~production.index.duplicated(keep="first")]
    weather_data = weather_data[~weather_data.index.duplicated(keep="first")]
    stats["production_duplicates_dropped"] = n_dup_production
    stats["weather_duplicates_dropped"] = n_dup_weather

    # (b) rate -> "energy if sustained for an hour" for the 15-min slices,
    # then resample 15-min -> hourly
    production = production.copy()
    production.loc[:, PRODUCTION_FLOW_COLS] = production[PRODUCTION_FLOW_COLS] * 0.25
    production_hourly = production.resample("h").agg(PRODUCTION_HOURLY_AGG)
    stats["production_hourly_rows_before_dst_fix"] = len(production_hourly)

    # (c) resample-created NaN-metadata rows (the "4 extra rows" quirk)
    nan_date_rows = production_hourly.index[production_hourly["Date"].isna()]
    production_hourly = production_hourly.dropna(subset=["Date"])
    stats["dropped_nan_metadata_rows"] = len(nan_date_rows)
    stats["dropped_nan_metadata_timestamps"] = [str(t) for t in nan_date_rows]

    # (d) DST / boundary alignment - generalised symmetric-difference fix
    # (see module docstring for why this differs from the notebook's
    # single hard-coded date pair)
    weather_only = weather_data.index.difference(production_hourly.index)
    production_only = production_hourly.index.difference(weather_data.index)
    weather_aligned = weather_data.drop(index=weather_only, errors="ignore")
    production_aligned = production_hourly.drop(index=production_only, errors="ignore")
    stats["weather_only_timestamps_dropped"] = [str(t) for t in weather_only]
    stats["production_only_timestamps_dropped"] = [str(t) for t in production_only]

    remaining_diff = production_aligned.index.symmetric_difference(weather_aligned.index)
    assert len(remaining_diff) == 0, f"Indices still differ: {remaining_diff}"

    production_aligned = production_aligned.sort_index()
    weather_aligned = weather_aligned.sort_index()

    # (e) merge
    merged = pd.concat([production_aligned, weather_aligned], axis=1)
    stats["merged_rows"] = len(merged)
    stats["merged_cols"] = merged.shape[1]

    return merged, stats


def main():
    production, fw_funchal, lpma = load_raw()
    weather_data = merge_weather(fw_funchal, lpma)
    merged, stats = merge_production_weather(production, weather_data)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PATH
    merged.to_csv(out_path)

    print("=== 01_merge_data.py ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Wrote merged hourly dataframe -> {out_path} shape={merged.shape}")
    return merged, stats


if __name__ == "__main__":
    main()
