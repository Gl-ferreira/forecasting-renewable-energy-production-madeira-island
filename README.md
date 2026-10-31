# Forecasting Renewable Energy Production in Madeira Island

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/paper-under%20review-orange)]()

A synthetic-data reproduction of the modeling pipeline behind the manuscript **"Forecasting Renewable Energy Production in Madeira Island,"** submitted to *Cleaner Energy Systems* (Elsevier).

### Contents

1. [Motivation](#1-motivation)
2. [Data](#2-data)
3. [Repository structure](#3-repository-structure)
4. [Pipeline](#4-pipeline)
5. [Results](#5-results)
6. [License](#6-license)
7. [Contact](#7-contact)

## 1. Motivation

Fossil fuels still cover most of the world's growing energy demand, and island grids feel the cost of that more than most. Madeira Island, a small Portuguese island, has no interconnector to a larger mainland grid, so every megawatt of hydropower, wind, solar, and biomass it brings online has to be balanced locally, in real time, against fossil and natural-gas backup. The island already has a meaningful renewable mix across all four of those sources; the bottleneck is predictability, not capacity. A grid operator who can forecast renewable output a day or two ahead can schedule storage and backup more precisely, cut fossil-fuel standby, and safely let more renewables onto the grid. Closing that gap with machine learning is what this project sets out to do.

This started as my master's thesis in Data Science and Advanced Analytics at NOVA IMS, built on real operational data from EEM (Empresa de Electricidade da Madeira), the island's utility, shared with me under a confidentiality arrangement. I built and compared four modeling approaches, LSTM, MLP, Random Forest, and XGBoost, across all four renewable sources. The results held up well enough, and felt useful enough beyond an academic exercise, that my supervisors, Maria Anastasiadou and Vítor Santos, and I decided to take it further and turn it into a peer-reviewed article. This repository is the part of that work I can make public.

The full thesis is open access if you want the longer version of this story: [NOVA RUN repository](https://run.unl.pt/entities/publication/21a7783f-52e1-4096-a626-2cc963cef5d9).

## 2. Data

None of the data under `data/` is real. `synthetic_production.csv` and the two `synthetic_weather_*.csv` files come out of `generate_synthetic_data.py`, a calibrated generator built to match the _structure and statistical properties_ of the real datasets (same shape, timestamps, column names, seasonality, diurnal patterns, skew, and missing-value patterns) without reproducing a single real record, row, or cell. Calibration used only aggregate statistics already public in the manuscript (installed capacity by year, seasonal shares) plus distributional properties profiled from the real data; no real values appear anywhere in this repo.

As a sanity check, here's how the synthetic monthly means track the real ones across all four sources:

![Synthetic vs. real seasonal profile](synthetic_vs_real_seasonal.png)

A consequence of all this: every result this pipeline produces (feature importances, R², RMSE, MAE, MAPE, SHAP values) is computed on synthetic data, and <u>won't match the numbers reported in the published manuscript</u>. The manuscript's real numbers come from the confidential EEM and weather data, which aren't published here or anywhere. This repo's job is narrower: let a reader verify that the modeling _methodology_ described in the manuscript is implemented as described and runs end to end, not let them reproduce the manuscript's actual findings.

Why isn't the real data here instead? Two reasons:

- **Production data**: it belongs to EEM, shared with me under a confidentiality arrangement for the thesis. EEM had no objection to the thesis becoming a journal article, but that didn't extend to publishing the underlying data or code.
- **Weather data**: pulled from the Visual Crossing Weather API. It isn't confidential, but its license prohibits republishing raw records in bulk, and the full historical pull used in the study is well beyond the API's free-tier query caps.

A synthetic, calibrated substitute sidesteps both problems while still letting the full pipeline run end to end for anyone who wants to inspect it. Each pipeline script's docstring names the exact section(s) of the original, private thesis notebook it reproduces, so the mapping between this code and the published methodology stays traceable.

## 3. Repository structure

```
synthetic_data_repo/
├── generate_synthetic_data.py        # Builds the 3 synthetic CSVs below from calibrated distributions
├── synthetic_vs_real_seasonal.png    # Validation chart: synthetic vs. real seasonal profile
├── data/
│   ├── synthetic_production.csv              # Synthetic EEM production, 15-min resolution
│   ├── synthetic_weather_FW_Funchal.csv       # Synthetic weather, Funchal station, hourly
│   ├── synthetic_weather_LPMA.csv             # Synthetic weather, LPMA station, hourly
│   └── processed/                             # Output of pipeline/01-04 (generated, not checked in blank)
│       ├── <target>_{train,val,test,trainval,test_tv}.csv
│       ├── <target>_xgb_importance.csv
│       ├── scalers/                           # Fitted scikit-learn scalers (joblib)
│       └── final/                             # Output of pipeline/05
│           └── <target>/{model.joblib, metrics.json, line_chart.png, shap_beeswarm.png}
├── real_results/                     # Real-data charts from the published manuscript (see section 5)
│   └── <target>/{line_chart.png, shap_beeswarm.png}
├── pipeline/
│   ├── 01_merge_data.py              # Merge production + weather
│   ├── 02_preprocess.py              # Cleaning, DST alignment, resampling
│   ├── 03_feature_engineering.py     # Lags, rolling windows, cyclical encodings
│   ├── 04_scale_and_select.py        # Scaling + XGBoost-based feature selection
│   ├── 05_train_final_models.py      # Final-round model training + SHAP
│   └── requirements.txt
└── colab/
    └── synthetic_data_repo_real_xgboost.ipynb   # Colab runner for 04 and 05 (see 4.2)
```

## 4. Pipeline

Run the scripts in order from the repo root:

```bash
pip install -r pipeline/requirements.txt

python3 generate_synthetic_data.py --out ./data
python3 pipeline/01_merge_data.py
python3 pipeline/02_preprocess.py
python3 pipeline/03_feature_engineering.py
python3 pipeline/04_scale_and_select.py
python3 pipeline/05_train_final_models.py
```

Each script reads the previous one's output from `data/` and writes its own output back there. The final step produces, for each of the four targets (Solar, Wind, Hydropower, Biomass), a trained model, its evaluation metrics, an actual-vs-predicted line chart, and a SHAP beeswarm summary plot, under `data/processed/final/<target>/`, plus a combined `data/processed/final/final_summary.csv`.

### 4.1 Models

The thesis evaluated LSTM, MLP, Random Forest, and XGBoost across two rounds of experiments. LSTM was eliminated after round 1 (weakest performer; not enough data for deep learning to pull ahead) and MLP was eliminated after round 2 (SHAP explanations are slow and only approximate for MLPs, versus fast and exact for tree-based models). The final round concluded that **XGBoost** works best for Hydropower and Wind, while **Random Forest** works best for Solar and Biomass, a hybrid solution, which `05_train_final_models.py` reproduces using the exact grid-search-winning hyperparameters from the source notebook.

### 4.2 Environment notes

`04_scale_and_select.py` and `05_train_final_models.py` use `xgboost` for feature selection and, for two of the four targets, for the final model. `05_train_final_models.py` also uses `shap` for the explainability plots. Both scripts fall back to pure scikit-learn equivalents (`GradientBoostingRegressor` in place of `XGBRegressor`, skipping the SHAP plot) if `xgboost`/`shap` aren't installed, and this always gets reported explicitly in the output (`xgboost_available`, `shap_available` flags) rather than failing silently. If you hit installation trouble with `xgboost`/`shap` in your own environment, `colab/synthetic_data_repo_real_xgboost.ipynb` is a ready-to-run Google Colab notebook that installs both and runs the full pipeline end to end. This is how the committed `data/processed/` and `data/processed/final/` results in this repo were actually produced.

### 4.3 A note on feature selection results

The feature importance lists in `data/processed/<target>_xgb_importance.csv` include `Fossil` and `Natural Gas` generation among the selected features for Wind and Biomass (and, just below Solar's threshold, for Solar too). This is a property of the synthetic data, not a physically meaningful relationship: the generator drives all production columns from broadly similar seasonal and calendar signals, which creates spurious cross-correlations between unrelated energy sources that wouldn't exist in, and aren't present in, the real data's feature-selection results reported in the manuscript.

### 4.4 Sanity check: pipeline output on synthetic data

The real data can't be shared (section 2), so there's no way to hand someone the actual EEM dataset to confirm this code works. Running it end to end on the included synthetic data instead at least proves it executes, each step produces output of the right shape, and the final step trains and evaluates a model. For reference, here's what that run produces (see `data/processed/final/final_summary.csv`). <u>These numbers carry no scientific meaning</u> — they're an artifact of the synthetic generator, not a result to compare against the real work; see section 5 for the real published metrics.

| Target | Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|---|
| Solar | Random Forest | 0.951 | 2.35 | 1.88 | n/a |
| Wind | XGBoost | 0.756 | 14.19 | 11.12 | 3.23% |
| Hydropower | XGBoost | 0.951 | 13.99 | 10.74 | 5.92% |
| Biomass | Random Forest | 0.876 | 2.68 | 2.06 | n/a |

<table>
<tr>
<td><img src="data/processed/final/solar/line_chart.png" width="400"></td>
<td><img src="data/processed/final/wind/line_chart.png" width="400"></td>
</tr>
<tr>
<td><img src="data/processed/final/hydropower/line_chart.png" width="400"></td>
<td><img src="data/processed/final/biomass/line_chart.png" width="400"></td>
</tr>
</table>

Each target's folder under `data/processed/final/` also has a SHAP beeswarm plot (`shap_beeswarm.png`) showing which features actually drove that model's predictions on the synthetic data.

## 5. Results

The final-round metrics reported in the manuscript (Table 4), trained and evaluated on the real, confidential EEM production and weather data described in section 2. <u>These are the actual published results</u> — not reproducible from this repo (section 2 explains why the real data isn't included); section 4.4 has a synthetic-data sanity check of the pipeline itself instead.

| RES | Model | MAE (MWh) | RMSE (MWh) | R² |
|---|---|---|---|---|
| Solar | Random Forest | 12.9 | 15.7 | 0.33 |
| Wind | XGBoost | 142.8 | 187.9 | 0.68 |
| Hydropower | XGBoost | 27.2 | 42.2 | 0.95 |
| Biomass | Random Forest | 8.8 | 16.9 | 0.75 |

<table>
<tr>
<td><img src="real_results/solar/line_chart.png" width="400"></td>
<td><img src="real_results/wind/line_chart.png" width="400"></td>
</tr>
<tr>
<td><img src="real_results/hydropower/line_chart.png" width="400"></td>
<td><img src="real_results/biomass/line_chart.png" width="400"></td>
</tr>
</table>

<table>
<tr>
<td><img src="real_results/solar/shap_beeswarm.png" width="400"></td>
<td><img src="real_results/wind/shap_beeswarm.png" width="400"></td>
</tr>
<tr>
<td><img src="real_results/hydropower/shap_beeswarm.png" width="400"></td>
<td><img src="real_results/biomass/shap_beeswarm.png" width="400"></td>
</tr>
</table>

## 6. License

The code in this repository (`generate_synthetic_data.py`, everything under `pipeline/`, and the Colab notebook) is released under the MIT License, see `LICENSE`. The synthetic datasets under `data/` are released under the same terms; use them freely, including for your own experiments or teaching, with the understanding that they're synthetic and don't represent real measurements.

## 7. Contact

If you use this code or the synthetic datasets, please cite the published manuscript once available:

> [Author names], "Forecasting Renewable Energy Production in Madeira Island," *Cleaner Energy Systems* (Elsevier), under review. Citation details will be added here once the article is formally accepted and published.

Questions about this repository can be directed to the corresponding author of the manuscript.
