"""
01_merge_data.py
=================
Reproduces notebook sections 1.1 (imports/initial checks), 1.2.1 (merging the two
weather datasets) and 1.2.2 (merging Production with weather), from
`COPY -Final code/preprocessing/data_preprocessing_and_modeling.ipynb`
(cells ~6-72), adapted to run on the synthetic CSV replacements for
Production.pkl / FW_Funchal.pkl / LPMA.pkl.

What this script does, in the same order as the original notebook:

1. Load the two synthetic hourly weather files and the synthetic 15-minute
   production file.
2. 1.2.1 - Weather merge: convert `winddir` (degrees) to sine/cosine components
   on BOTH stations (a true circular mean cannot be taken on raw degrees),
   drop the original degree/radian columns, then average the two stations'
   remaining numeric columns (including winddir_sin/winddir_cos) with a plain
   arithmetic mean -> `weather_data`. This is exactly what the original
   notebook does (it does not renormalise the sin/cos pair back onto the unit
   circle after averaging - replicated as-is).
3. 1.2.2 - Production/weather merge:
   a. Drop duplicate timestamps on both datasets, keeping the first record.
   b. Production values are "rate if sustained for an hour", so energy
      columns are first multiplied by 0.25 (15-min -> quarter of an hour)
      and then the 15-min data is resampled to hourly bins: flow/energy
      columns via `sum`, descriptive/metadata columns via `first`.
   c. This resampling step is known (per the original notebook's own
      commentary) to silently create extra hourly rows with NaN metadata
      whenever an entire hour has no underlying 15-min rows (e.g. the
      "missing hour" at a spring-forward DST transition) - these are
      dropped via `dropna(subset=['Date'])`. The same step also means an
      all-NaN hour sums to 0.0 instead of NaN for columns aggregated with
      `sum` (notably Hydropower during its real missingness window,
      November 2021); this is NOT patched here, exactly as in the original,
      because section 2.3's Random Forest imputer step explicitly treats
      `isna() | eq(0)` as "missing" for Hydropower, downstream.
   d. DST alignment: in the original (real) data the spring-forward "gap"
      hour landed on 26 March in the weather feed but 28 March in the
      production feed for 2023, so the mismatched single hour was dropped
      from each dataset. In the SYNTHETIC data, the production feed
      reproduces a similar (but not identical) spring-forward gap in EVERY
      year of the series, while the synthetic weather feed has no DST gap
      at all (it's a perfectly continuous hourly series). See the
      DEVIATION note below for how this was adapted.
   e. The two frames are concatenated on columns (axis=1) once their
      indices match exactly.

DEVIATIONS from the original notebook (synthetic-data-specific judgment
calls - see the task report for detail):
  - The original notebook hard-codes a single date-specific drop
    (`'2023-03-26 01:00:00'` from Production, `'2023-03-28 01:00:00'` from
    weather) because that was the one mismatch found in the real data. The
    synthetic data instead has a reproducible spring-forward gap in
    Production for every covered year (2021-03-28, 2022-03-27, 2023-03-28,
    2024-03-31), and no gap at all in weather. This script therefore
    generalises the ORIGINAL INTENT ("drop whichever single hour exists in
    one dataset but not the other around the DST transition") rather than
    the original's literal hard-coded dates: after fixing the production
    side (dropna on the synthetic NaN-metadata rows), it computes the
    symmetric difference between the two indices and drops whatever is
    left over on each side. This reduces to the identical fix, with the
    identical row-count effect, if the synthetic data behaved exactly like
    the real data.
  - The synthetic production series extends 1 hour later on its last day
    (2024-08-31 23:00) than the synthetic weather series (whose last hourly
    reading is 22:00) -- an artefact of the two generators' end-of-range
    boundaries, not a DST issue. That single trailing hour is dropped by the
    same symmetric-difference step above, instead of being left as a
    NaN-weather row in the merged frame.

Output: `data/merged_hourly.csv` - the single hourly dataframe `df` that
notebook cell 72 (`pd.concat([Production, weather_data], axis=1)`) produces,
ready for section 2 (`02_preprocess.py`).
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
