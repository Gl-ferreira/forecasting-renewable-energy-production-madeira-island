"""
05_train_final_models.py
=========================

Step 5, the payoff: trains the final model for each of the four renewable
sources and explains its predictions with SHAP, so the output is something
a grid operator could actually trust, not just a black-box number.
Reproduces notebook sections 3.3 (Final Round) and 3.4 (Proposal of the best
model), cells ~207-222 of `data_preprocessing_and_modeling.ipynb`
(`COPY -Final code/preprocessing/`, read-only), on the synthetic data.

WHICH MODEL PER TARGET (do not change without re-reading the source notebook):
Cell 190: LSTM was eliminated after round 1 ("the worst model" - deep
learning needs more data than this study has). Cell 206: MLP was eliminated
after round 2 (SHAP is slow/approximate for MLPs, fast/exact for tree
models). Cell 220, verbatim: "XGBoost was the top performer for Hydropower
and Wind, whereas Random Forest performed better for Solar and Biomass.
Therefore, the optimal solution is a hybrid one, using XGBoost for
Hydropower and Wind, and Random Forest for Solar and Biomass." That is the
published methodology this script reproduces - NOT the stray
`lstm_daily_lb14_final.h5` files found under `modeling/final/solar` and
`modeling/final/wind` on disk, which appear to be an earlier/abandoned
experiment and contradict the notebook's own narrative text. The notebook's
markdown conclusion is the source of truth here, not leftover files.

Hyperparameters (grid-search winners, verbatim from cells 213 and 217):
  XGBoost:
    wind:       n_estimators=100, max_depth=3, learning_rate=0.05
    hydropower: n_estimators=200, max_depth=3, learning_rate=0.05
  Random Forest:
    solar:   n_estimators=120, max_depth=8,  min_samples_split=6
    biomass: n_estimators=100, max_depth=10, min_samples_split=4
  Both: random_state=42, n_jobs=-1 (matching the notebook's defaults).

For each of the 4 targets, this script:
  1. Loads `<target>_trainval.csv` / `<target>_test_tv.csv` from
     `data/processed/` (output of 04_scale_and_select.py - already reduced
     to the real-XGBoost-selected feature set for that target).
  2. Trains the model the notebook selected for that target, on trainval.
  3. Predicts on test_tv, inverse-transforms to original units (MWh) via
     the target's round2_trainval y-scaler: `scaler_y.inverse_transform`
     then `np.square` (undoing the sqrt + MinMax from 04_scale_and_select.py),
     exactly as cells 212/216 do.
  4. Computes RMSE/MAE/R2 (XGBoost also computes MAPE/NMAE/NRMSE, matching
     cell 212's extra metrics - the original RF cell, 216, only computes
     RMSE/MAE/R2, so this script does the same per-model-type split).
  5. Runs SHAP TreeExplainer (works for both RandomForestRegressor and
     XGBRegressor/GradientBoostingRegressor) and saves a beeswarm summary
     plot, plus an actual-vs-predicted line chart - same two visualization
     types cells 212/216 produce.
  6. Saves the fitted model (joblib), metrics (json), and both plots under
     `data/processed/final/<target_lower>/`.

ENVIRONMENT NOTE: same as 04_scale_and_select.py - if real `xgboost` is not
installed, this script falls back to `sklearn.ensemble.GradientBoostingRegressor`
for the two XGBoost-selected targets (Wind, Hydropower), clearly flagged via
`stats["xgboost_available"]`, never silent. Random Forest (Solar, Biomass)
is scikit-learn either way, so those two results do not depend on xgboost
availability.

Output: data/processed/final/<target_lower>/model.joblib
        data/processed/final/<target_lower>/metrics.json
        data/processed/final/<target_lower>/line_chart.png
        data/processed/final/<target_lower>/shap_beeswarm.png
        data/processed/final/final_summary.csv  (all 4 targets, one row each)
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    _HAS_XGBOOST = False

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SCALER_DIR = PROCESSED_DIR / "scalers"
FINAL_DIR = PROCESSED_DIR / "final"

TARGETS = ["Solar", "Wind", "Hydropower", "Biomass"]

MODEL_CHOICE = {
    "Hydropower": "xgb",
    "Wind": "xgb",
    "Solar": "rf",
    "Biomass": "rf",
}

XGB_PARAMS = {
    "Wind": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    "Hydropower": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
}

RF_PARAMS = {
    "Solar": {"n_estimators": 120, "max_depth": 8, "min_samples_split": 6},
    "Biomass": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 4},
}

RANDOM_STATE = 42
COLOR_PRED = {"Solar": "red", "Wind": "orange", "Hydropower": "blue", "Biomass": "green"}


def load_trainval_test(target: str):
    name = target.lower()
    trainval = pd.read_csv(PROCESSED_DIR / f"{name}_trainval.csv", index_col=0, parse_dates=True)
    test_tv = pd.read_csv(PROCESSED_DIR / f"{name}_test_tv.csv", index_col=0, parse_dates=True)

    X_trainval = trainval.drop(columns=[target])
    y_trainval = trainval[target]
    X_test = test_tv.drop(columns=[target])
    y_test = test_tv[target]
    return X_trainval, y_trainval, X_test, y_test


def build_model(target: str):
    choice = MODEL_CHOICE[target]
    if choice == "xgb":
        params = dict(XGB_PARAMS[target])
        params.setdefault("random_state", RANDOM_STATE)
        if _HAS_XGBOOST:
            model = XGBRegressor(n_jobs=-1, **params)
        else:
            # Fallback: GradientBoostingRegressor doesn't take n_jobs/learning_rate
            # the same way XGBRegressor's full surface does, but accepts the core
            # three hyperparameters directly.
            model = GradientBoostingRegressor(**params)
        return model, choice, params
    else:
        params = dict(RF_PARAMS[target])
        params.setdefault("random_state", RANDOM_STATE)
        model = RandomForestRegressor(n_jobs=-1, **params)
        return model, choice, params


def inverse_to_original_units(y_scaled: np.ndarray, scaler_y) -> np.ndarray:
    """Undo round2_trainval's sqrt + MinMax scaling: inverse MinMax, then square."""
    y_sqrt = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
    return np.square(y_sqrt).reshape(-1)


def train_and_evaluate(target: str, stats: dict):
    name = target.lower()
    out_dir = FINAL_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_trainval, y_trainval, X_test, y_test = load_trainval_test(target)
    model, choice, params = build_model(target)

    model.fit(X_trainval.to_numpy(dtype=np.float32), y_trainval.to_numpy(dtype=np.float32))

    y_pred_scaled = model.predict(X_test.to_numpy(dtype=np.float32))

    scaler_y = joblib.load(SCALER_DIR / f"scaler_y_{name}_round2_trainval.pkl")
    y_pred = inverse_to_original_units(y_pred_scaled, scaler_y)
    y_true = inverse_to_original_units(y_test.to_numpy(dtype=np.float32), scaler_y)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {"model": choice, "params": params, "rmse": rmse, "mae": mae, "r2": r2}

    if choice == "xgb":
        mean_actual = float(np.mean(y_true))
        nonzero_mask = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
            if nonzero_mask.any()
            else None
        )
        nmae = mae / mean_actual if mean_actual != 0 else None
        nrmse = rmse / mean_actual if mean_actual != 0 else None
        metrics.update({"mape": mape, "nmae": nmae, "nrmse": nrmse})

    # Line chart: actual vs predicted on test
    plt.figure(figsize=(10, 3))
    plt.plot(X_test.index, y_true, label="Actual", linewidth=1.8, color="black")
    plt.plot(
        X_test.index,
        y_pred,
        label="Predicted",
        linewidth=1.8,
        linestyle="--",
        color=COLOR_PRED[target],
    )
    plt.title(f"{target} ({choice.upper()})", fontsize=16, fontweight="bold")
    plt.ylabel("MWh", fontsize=12, fontweight="bold")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "line_chart.png", dpi=150)
    plt.close()

    shap_ok = False
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.to_numpy(dtype=np.float32))
            plt.figure()
            shap.summary_plot(
                shap_values,
                X_test,
                max_display=20,
                show=False,
                plot_type="dot",
            )
            plt.title(f"{target} ({choice.upper()}) - SHAP", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close()
            shap_ok = True
        except Exception as e:  # pragma: no cover - diagnostic path
            metrics["shap_error"] = str(e)
    metrics["shap_plot_saved"] = shap_ok

    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(
        f"[{target.upper()}] {choice.upper()} trainval->test | RMSE: {rmse:,.4f} | "
        f"MAE: {mae:,.4f} | R2: {r2:,.4f} | params: {params} | shap_saved: {shap_ok}"
    )

    stats[target] = metrics
    return metrics


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats["xgboost_available"] = _HAS_XGBOOST
    stats["shap_available"] = _HAS_SHAP

    summary_rows = []
    for target in TARGETS:
        metrics = train_and_evaluate(target, stats)
        row = {"target": target, "model": metrics["model"], "rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]}
        if "mape" in metrics:
            row.update({"mape": metrics["mape"], "nmae": metrics["nmae"], "nrmse": metrics["nrmse"]})
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("target")
    summary_df.to_csv(FINAL_DIR / "final_summary.csv")

    print("=== 05_train_final_models.py ===")
    print(f"xgboost_available: {_HAS_XGBOOST}")
    print(f"shap_available: {_HAS_SHAP}")
    print(summary_df)

    return stats, summary_df


if __name__ == "__main__":
    main()
