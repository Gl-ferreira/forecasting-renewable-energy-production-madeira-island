"""
Synthetic data generator for the Madeira RES forecasting portfolio repo.

Madeira can't lean on a mainland interconnector, so getting renewable
production forecasts right matters more here than almost anywhere else in
Europe. This script exists so that part of the story can be reproduced and
run by anyone, without me having to hand over EEM's real, confidential data.

Produces data that is STRUCTURALLY AND STATISTICALLY SIMILAR to the real,
proprietary EEM production dataset and the Visual Crossing weather dataset
used in the published study — same shape, timestamps, column names, rough
seasonality/diurnal patterns, skew, and missing-value patterns — but every
value is synthetically generated. No real record, row, or cell from the
original datasets is reproduced anywhere in this script or its output.

Calibration targets (profiled read-only from the real data on 2026-10-16,
not reproduced here beyond aggregate statistics already consistent with
what is publicly stated in the published manuscript, e.g. seasonal shares,
installed capacity, variable lists, and skewness/outlier/missingness rates):
  - Production: 128,540 rows, 15-min resolution, 2021-01-01..2024-08-31,
    6 energy sources + capacities, Hydropower missing all of Nov 2021,
    DST duplicate/missing timestamps on the documented dates.
  - Weather: 32,135 rows per station, hourly, 16 variables, windgust ~18.5%
    missing, visibility/solarradiation/solarenergy/uvindex ~0.03% missing.

Run: python3 generate_synthetic_data.py --out ./data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Production (EEM) synthetic generator
# ---------------------------------------------------------------------------

# Real per-year installed capacities (already public — reported in the
# published manuscript's Fig. 2, not proprietary production records).
CAPACITY_BY_YEAR = {
    2021: dict(Wind=63.11, Hydropower=79.0, Solar=15, Biomass=9, Fossil=170.0, NaturalGas=56),
    2022: dict(Wind=63.10, Hydropower=79.0, Solar=15, Biomass=9, Fossil=170.0, NaturalGas=56),
    2023: dict(Wind=63.10, Hydropower=79.0, Solar=15, Biomass=9, Fossil=170.0, NaturalGas=56),
    2024: dict(Wind=63.10, Hydropower=82.0, Solar=16, Biomass=9, Fossil=168.0, NaturalGas=56),
}

# Monthly seasonal shape (relative weights, not real values) derived from
# the documented seasonal pattern: hydropower peaks Nov-Mar, dips Jun-Sep;
# solar peaks in summer; wind higher in winter.
HYDRO_MONTH_SHAPE = {1: 1.5, 2: 1.7, 3: 1.75, 4: 1.35, 5: 0.55, 6: 0.6,
                      7: 0.25, 8: 0.2, 9: 0.28, 10: 0.45, 11: 0.42, 12: 1.4}
SOLAR_MONTH_SHAPE = {1: 0.55, 2: 0.7, 3: 0.85, 4: 0.95, 5: 1.0, 6: 1.0,
                      7: 1.1, 8: 1.05, 9: 0.82, 10: 0.8, 11: 0.62, 12: 0.52}
WIND_MONTH_SHAPE = {1: 1.0, 2: 1.08, 3: 1.0, 4: 0.76, 5: 0.94, 6: 0.95,
                     7: 0.64, 8: 0.68, 9: 0.79, 10: 0.75, 11: 0.64, 12: 1.07}
BIOMASS_MONTH_SHAPE = {1: 1.0, 2: 0.93, 3: 0.69, 4: 0.98, 5: 1.05, 6: 0.84,
                        7: 0.7, 8: 0.89, 9: 1.0, 10: 0.57, 11: 0.59, 12: 0.98}

# Diurnal shape for solar (0 at night, bell curve daytime) and a mild
# inverse diurnal shape for wind (slightly higher overnight).
HOURS = np.arange(24)
SOLAR_DIURNAL = np.clip(np.sin((HOURS - 6) / 12 * np.pi), 0, None) ** 1.3
WIND_DIURNAL = 1.0 - 0.18 * np.sin((HOURS - 13) / 12 * np.pi)


def skewed_series(n, mean, std, skew, rng, floor=None):
    """Generate a right- or left-skewed series with the requested mean/std,
    using a shifted/scaled gamma (right skew) or mirrored gamma (left skew).
    floor=None leaves the series unclipped (use this for a zero-centred
    noise term meant to be added to a separate deterministic base — clip
    only the final combined series, not the noise itself)."""
    skew = np.clip(skew, -6, 6)
    if abs(skew) < 0.05:
        x = rng.normal(mean, std, n)
    else:
        shape = max(4.0 / (skew ** 2), 0.3)
        scale = std / np.sqrt(shape)
        g = rng.gamma(shape, scale, n)
        g = (g - g.mean()) / g.std() * std
        if skew < 0:
            g = -g
        x = mean + g
    if floor is not None:
        x = np.clip(x, floor, None)
    return x


def build_production(start="2021-01-01", end="2024-08-31 23:45", freq="15min"):
    idx = pd.date_range(start, end, freq=freq)

    # Apply the same DST pattern documented in the manuscript/real data:
    # duplicate the 01:00-01:45 hour on the last Sunday of October (fall
    # back), remove it on the last Sunday of March (spring forward).
    dst_fall = ["2021-10-31", "2022-10-30", "2023-10-31"]
    dst_spring = ["2021-03-28", "2022-03-27", "2023-03-28", "2024-03-31"]

    spring_drop = pd.DatetimeIndex(
        [pd.Timestamp(f"{d} {h:02d}:{m:02d}") for d in dst_spring for h in [1] for m in (0, 15, 30, 45)]
    )
    idx = idx.difference(spring_drop)

    fall_dup = pd.DatetimeIndex(
        [pd.Timestamp(f"{d} {h:02d}:{m:02d}") for d in dst_fall for h in [1] for m in (0, 15, 30, 45)]
    )
    idx = idx.append(fall_dup).sort_values()

    n = len(idx)
    months = idx.month
    hours = idx.hour
    years = idx.year

    def month_weight(shape_dict):
        return np.array([shape_dict[m] for m in months])

    def hour_weight(diurnal):
        return diurnal[hours]

    # --- Hydropower: mean ~8.74, skew ~1.38. Seasonal base (mean ~0) plus
    # unclipped skewed noise, clipped only once combined.
    hydro_month = month_weight(HYDRO_MONTH_SHAPE)
    hydro_base = 4.3 * (hydro_month - hydro_month.mean())
    hydropower = np.clip(8.74 + hydro_base + skewed_series(n, 0, 4.6, 1.38, RNG), 0, 59)

    # --- Solar: diurnal + seasonal, skew ~1.17, near-zero at night.
    solar_month = month_weight(SOLAR_MONTH_SHAPE) / np.mean(list(SOLAR_MONTH_SHAPE.values()))
    solar_diurnal = hour_weight(SOLAR_DIURNAL)
    solar = (9.5 * solar_month * solar_diurnal
             + skewed_series(n, 0, 2.2, 1.17, RNG) * solar_diurnal)
    solar = np.clip(solar, 0, 15.3)

    # --- Wind: seasonal + mild diurnal, skew ~0.76.
    wind_month = month_weight(WIND_MONTH_SHAPE)
    wind_base = 3.0 * (wind_month - wind_month.mean())
    wind = np.clip(15.1 + wind_base * hour_weight(WIND_DIURNAL) + skewed_series(n, 0, 6.2, 0.76, RNG), 0, 59)

    # --- Biomass: fairly flat, left-skewed (-1.38), bounded near capacity.
    biomass_month = month_weight(BIOMASS_MONTH_SHAPE)
    biomass_base = 0.6 * (biomass_month - biomass_month.mean())
    biomass = np.clip(4.7 + biomass_base + skewed_series(n, 0, 0.95, -1.38, RNG), 0, 7.3)

    # --- Fossil & Natural Gas: largely complementary to renewables (grid
    # balancing), giving the generator roughly the right inverse
    # correlation with wind/hydropower without hand-tuning a copula.
    renew_total = hydropower + wind + solar + biomass
    fossil = np.clip(84.0 - 1.05 * renew_total + skewed_series(n, 0, 7.0, -0.04, RNG), 5, 120)
    natgas = np.clip(17.3 + skewed_series(n, 0, 5.8, 0.18, RNG), 0, 50)
    curtailed = np.clip(skewed_series(n, 0, 1.3, 6.0, RNG, floor=0.0) * (wind > 40).astype(float) * RNG.random(n) * 3, 0, 46)

    df = pd.DataFrame({
        "Date": idx.normalize(),
        "Hydropower": hydropower,
        "Fossil": fossil,
        "Natural Gas": natgas,
        "Biomass": biomass,
        "Wind": wind,
        "Solar": solar,
        "Curtailed Wind Energy": curtailed,
        "Year": years,
    }, index=idx)
    df.index.name = "Hora"

    for col, key in [("Wind Energy Capacity", "Wind"), ("Hydropower Energy Capacity", "Hydropower"),
                      ("Solar Energy Capacity", "Solar"), ("Biomass Energy Capacity", "Biomass"),
                      ("Fossil Energy Capacity", "Fossil"), ("Natural Gas Energy Capacity", "NaturalGas")]:
        df[col] = df["Year"].map(lambda y: CAPACITY_BY_YEAR[y][key])

    df["Month Name"] = df.index.month_name()
    df["Day of Month"] = df.index.day
    df["Month"] = df.index.month
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["Season"] = df["Month"].map(season_map)
    df["Hour of the day"] = df.index.hour
    year_start = df.index.to_series().dt.to_period("Y").dt.start_time
    df["Hour of the Year"] = ((df.index.to_series() - year_start).dt.total_seconds() // 3600).astype(int)

    # Inject the Nov-2021 Hydropower gap (2,880 rows), matching the real
    # dataset's documented missing-value pattern exactly in shape.
    nov_mask = (df.index.year == 2021) & (df.index.month == 11)
    df.loc[nov_mask, "Hydropower"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Weather synthetic generator
# ---------------------------------------------------------------------------

WEATHER_STATION_PARAMS = {
    "FW_Funchal": dict(windspeed_mean=8.94, windspeed_std=4.69, windgust_mean=32.9, windgust_std=13.28,
                        humidity_mean=66.2, humidity_std=9.56, pressure_mean=1020.2),
    "LPMA": dict(windspeed_mean=17.23, windspeed_std=9.6, windgust_mean=29.99, windgust_std=11.96,
                 humidity_mean=68.56, humidity_std=10.54, pressure_mean=1019.9),
}


def build_weather(station, n=32135, start="2021-01-01"):
    idx = pd.date_range(start, periods=n, freq="h")
    p = WEATHER_STATION_PARAMS[station]
    months = idx.month
    hours = idx.hour

    # Seasonal temperature: low thermic amplitude (17-22C) per the manuscript.
    temp_season = 19.5 + 2.2 * np.cos((months - 8) / 12 * 2 * np.pi)
    temp = temp_season + skewed_series(n, 0, 2.2, 0.16, RNG, floor=-15)
    feelslike = temp + RNG.normal(0, 0.3, n)

    humidity = np.clip(skewed_series(n, p["humidity_mean"], p["humidity_std"], -0.69, RNG, floor=0), 5, 100)
    dew = temp - (100 - humidity) / 5 + RNG.normal(0, 1, n)

    precip = skewed_series(n, 0.07, 1.2, 48, RNG, floor=0)
    precip = np.where(RNG.random(n) < 0.85, 0.0, precip)  # mostly dry hours
    precipprob = np.where(precip > 0, RNG.integers(20, 100, n), 0)
    snowdepth = np.zeros(n)

    wind_diurnal = hour_weight = 1.0 - 0.1 * np.sin((hours - 13) / 12 * np.pi)
    windspeed = np.clip(skewed_series(n, p["windspeed_mean"], p["windspeed_std"], 1.49, RNG, floor=0) * wind_diurnal, 0, 60)
    windgust = np.clip(windspeed * 1.0 + skewed_series(n, p["windgust_mean"] - p["windspeed_mean"], 10, 0.9, RNG, floor=0), 0, 125)
    winddir = RNG.uniform(0, 360, n)

    pressure = skewed_series(n, p["pressure_mean"], 4.6, -0.38, RNG, floor=990)
    visibility = np.clip(skewed_series(n, 11.0, 2.8, 1.0, RNG, floor=0), 0, 22)
    cloudcover = np.clip(skewed_series(n, 48, 22, 0.5, RNG, floor=0), 0, 100)

    solar_diurnal = SOLAR_DIURNAL[hours]
    month_solar_factor = np.array([SOLAR_MONTH_SHAPE[m] for m in months])
    solarradiation = np.clip(560 * solar_diurnal * month_solar_factor / max(SOLAR_MONTH_SHAPE.values())
                              + skewed_series(n, 0, 55, 1.8, RNG) * solar_diurnal, 0, 1050)
    solarenergy = np.clip(solarradiation / 278, 0, 3.8)
    uvindex = np.clip(np.round(solarradiation / 101), 0, 10)

    df = pd.DataFrame({
        "temp": temp, "feelslike": feelslike, "humidity": humidity, "dew": dew,
        "precip": precip, "precipprob": precipprob, "snowdepth": snowdepth,
        "windgust": windgust, "windspeed": windspeed, "winddir": winddir,
        "pressure": pressure, "visibility": visibility, "cloudcover": cloudcover,
        "solarradiation": solarradiation, "solarenergy": solarenergy, "uvindex": uvindex,
    }, index=idx)
    df.index.name = "date_time"

    # Inject missingness matching the documented rates (MCAR, for simplicity).
    for col, rate in [("windgust", 0.1848), ("visibility", 0.0001), ("solarradiation", 0.0003),
                       ("solarenergy", 0.0003), ("uvindex", 0.0003)]:
        mask = RNG.random(n) < rate
        df.loc[mask, col] = np.nan

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prod = build_production()
    prod.to_csv(out / "synthetic_production.csv")
    print("production:", prod.shape, "->", out / "synthetic_production.csv")

    for station in WEATHER_STATION_PARAMS:
        w = build_weather(station)
        w.to_csv(out / f"synthetic_weather_{station}.csv")
        print(station, ":", w.shape, "->", out / f"synthetic_weather_{station}.csv")


if __name__ == "__main__":
    main()
