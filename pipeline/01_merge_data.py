"""
01_merge_data.py

Step 1 of the pipeline: lines up two weather stations and the production
meter readings on a common hourly clock. Nothing clever yet, but getting
this join right is what every later feature and model depends on.

Maps to notebook sections 1.1, 1.2.1 and 1.2.2 of
`COPY -Final code/preprocessing/data_preprocessing_and_modeling.ipynb`
(roughly cells 6-72), adapted to run on the synthetic CSVs instead of
Production.pkl / FW_Funchal.pkl / LPMA.pkl. Same order as the notebook:

1. Load the two synthetic hourly weather files and the synthetic 15-minute
   production file.
2. Weather merge (1.2.1): convert `winddir` (degrees) to sine/cosine on
   both stations, since you can't take a circular mean on raw degrees, drop
   the original degree/radian columns, then average the two stations'
   remaining numeric columns with a plain arithmetic mean. One quirk kept
   on purpose: the notebook never renormalises the sin/cos pair back onto
   the unit circle after averaging, so neither does this.
3. Production/weather merge (1.2.2): drop duplicate timestamps on both
   sides, keeping the first record. Production values are a rate ("if
   sustained for an hour"), so the energy columns get multiplied by 0.25
   before resampling 15-min data up to hourly bins (sum for flow/energy
   columns, first for metadata). That resample step quietly creates a
   handful of extra hourly rows with NaN metadata whenever a whole hour has
   no underlying 15-min rows, most notably the missing hour at a
   spring-forward DST transition, and those get dropped. It also means an
   all-NaN hour sums to 0.0 instead of NaN for the `sum`-aggregated columns
   (this bites Hydropower during its real missingness window, November
   2021). Left as-is here, same as the notebook, because the imputer step
   in 02_preprocess.py already treats `isna() | eq(0)` as "missing" for
   Hydropower, so it gets handled downstream rather than patched twice.
   Finally, the two frames get aligned on timestamps that exist in both and
   concatenated column-wise.

One real adaptation versus the notebook, worth flagging: the real data had
exactly one DST mismatch (26 March in weather vs. 28 March in production,
2023 only), so the notebook just hard-codes dropping those two specific
timestamps. The synthetic generator reproduces a spring-forward gap in
production every year but has no DST gap in weather at all, so a single
hard-coded date pair wouldn't cover it. This script generalises the intent
instead: drop whichever hour exists on one side but not the other around
each DST transition, computed as a symmetric difference between the two
indices rather than listed by hand. Same effect, same row count, if the
synthetic data happened to match the real pattern exactly. It also happens
to clean up one more mismatch for free: the synthetic production series
runs one hour later on its last day than the synthetic weather series, an
artefact of how the two generators pick their end boundary, not a DST
issue, and that trailing hour gets caught by the same step.

Output: `data/merged_hourly.csv`, ready for `02_preprocess.py`.
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
