"""
04_scale_and_select.py

Step 4: scales everything onto comparable ranges and lets XGBoost vote on
which features earn a place in the final models, rather than including
every variable and hoping for the best. Reproduces notebook sections 2.6
(scaling) and 2.7 (feature selection), roughly cells 141-173.

Scaling (2.6) applies a sqrt transform first, to every numeric feature
column except the four targets (Solar, Wind, Hydropower, Biomass) and any
cyclical column ending in '_sin'/'_cos', then MinMax scaling on top. Two
rounds are built, matching the original: round 1 fits a MinMaxScaler on
train only and applies it to train/val/test, producing X_train_<target>,
X_val_<target>, X_test_<target> plus each target's own sqrt+MinMax y
(scaler fit on that target's train y only). Round 2 concatenates train and
val into `trainval`, refits the X scaler and each target's y scaler on
that, and applies them to trainval/test, producing X_trainval_<target>,
X_test_<target>_tv and the matching y_trainval/y_test_tv pairs. Every
target starts from the same full scaled X copy; features only get pared
down per target in 2.7. All of the scalers, X and each target's y, for both
rounds, are saved to `data/processed/scalers/` via joblib (the original
saves the same thing under modeling/round1 and modeling/round2_trainval),
so results can be de-scaled later.

Feature selection (2.7) first drops, for each target, any column with zero
variance on that target's X_train, from both X_train and X_test (round 1).
The `mutual_info_scores` and `xgboost_feature_importance` diagnostics are
ported verbatim from the notebook, same model hyperparameters throughout
(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8, importance_type="gain"). For the final per-target
list (Solar, Wind, Hydropower, Biomass in turn), the original notebook read
`xgboost_feature_importance(..., rel_thresh=1.0)`'s printed table by eye
and hard-coded whatever came out above roughly 1% relative importance.
Those exact, real-data-derived lists have no reason to still make sense on
synthetic data's different feature/target relationships, so this script
automates the same rule instead of reusing the hard-coded lists: for each
target, the final feature set is just whatever
`xgboost_feature_importance(X_train_target, y_train_target,
importance_type="gain", rel_thresh=1.0)` returns, the same 1% threshold the
notebook used to build the table it then read by eye. The run report has
the resulting feature lists and how they compare to the original hard-coded
ones.

A note on the environment: this script imports `xgboost.XGBRegressor`,
same as the original. If xgboost isn't installed, it falls back to
`sklearn.ensemble.GradientBoostingRegressor` with whatever core
hyperparameters it can accept, using its built-in `.feature_importances_`
(impurity-based; a reasonable but not identical stand-in for XGBoost's
"gain" importance). That fallback is always reported in the output rather
than happening silently. Install xgboost and re-run for the real thing.

Input:  data/train_daily.csv, data/val_daily.csv, data/test_daily.csv
Output: data/processed/<target>_train.csv, <target>_val.csv, <target>_test.csv
        data/processed/<target>_trainval.csv, <target>_test_tv.csv
        data/processed/scalers/*.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    _HAS_XGBOOST = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SCALER_DIR = PROCESSED_DIR / "scalers"

TARGETS = ["Solar", "Wind", "Hydropower", "Biomass"]
SOURCES = [t.lower() for t in TARGETS]


def load_daily_splits():
    train = pd.read_csv(DATA_DIR / "train_daily.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(DATA_DIR / "val_daily.csv", index_col=0, parse_dates=True)
    test = pd.read_csv(DATA_DIR / "test_daily.csv", index_col=0, parse_dates=True)
    return train, val, test


def sqrt_input_columns(train: pd.DataFrame):
    numeric_cols = train.select_dtypes(include="number").columns.tolist()
    cyclical_mask = [c.endswith("_sin") or c.endswith("_cos") for c in numeric_cols]
    targets_mask = [c in TARGETS for c in numeric_cols]
    return [
        c
        for c, is_cyc, is_tgt in zip(numeric_cols, cyclical_mask, targets_mask)
        if (not is_cyc) and (not is_tgt)
    ]


def scale_round1(train, val, test, cols_to_sqrt):
    """Notebook cell 141 - fit MinMax on TRAIN, apply to train/val/test;
    identical copies of X per target source."""
    train_proc, val_proc, test_proc = train.copy(), val.copy(), test.copy()
    for df in (train_proc, val_proc, test_proc):
        df[cols_to_sqrt] = np.sqrt(df[cols_to_sqrt])

    X_train_all = train_proc.drop(columns=TARGETS).select_dtypes(include="number")
    X_val_all = val_proc.drop(columns=TARGETS).select_dtypes(include="number")
    X_test_all = test_proc.drop(columns=TARGETS).select_dtypes(include="number")
    assert list(X_train_all.columns) == list(X_val_all.columns) == list(X_test_all.columns)

    scaler_X = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler_X.fit_transform(X_train_all), columns=X_train_all.columns, index=X_train_all.index
    )
    X_val_scaled = pd.DataFrame(
        scaler_X.transform(X_val_all), columns=X_val_all.columns, index=X_val_all.index
    )
    X_test_scaled = pd.DataFrame(
        scaler_X.transform(X_test_all), columns=X_test_all.columns, index=X_test_all.index
    )

    y = {}
    scalers_y = {}
    for target in TARGETS:
        name = target.lower()
        y_train_sqrt = np.sqrt(train_proc[[target]])
        y_val_sqrt = np.sqrt(val_proc[[target]])
        y_test_sqrt = np.sqrt(test_proc[[target]])

        scaler_y = MinMaxScaler()
        y[f"y_train_{name}"] = pd.DataFrame(
            scaler_y.fit_transform(y_train_sqrt), columns=[target], index=y_train_sqrt.index
        )
        y[f"y_val_{name}"] = pd.DataFrame(
            scaler_y.transform(y_val_sqrt), columns=[target], index=y_val_sqrt.index
        )
        y[f"y_test_{name}"] = pd.DataFrame(
            scaler_y.transform(y_test_sqrt), columns=[target], index=y_test_sqrt.index
        )
        scalers_y[name] = scaler_y

    X = {}
    for src in SOURCES:
        X[f"X_train_{src}"] = X_train_scaled.copy()
        X[f"X_val_{src}"] = X_val_scaled.copy()
        X[f"X_test_{src}"] = X_test_scaled.copy()

    return X, y, scaler_X, scalers_y


def scale_round2_trainval(train, val, test, cols_to_sqrt):
    """Notebook cell 142 - fit MinMax on TRAIN+VAL, apply to trainval/test."""
    trainval = pd.concat([train, val], axis=0).sort_index()
    trainval_proc, test_proc = trainval.copy(), test.copy()
    trainval_proc[cols_to_sqrt] = np.sqrt(trainval_proc[cols_to_sqrt])
    test_proc[cols_to_sqrt] = np.sqrt(test_proc[cols_to_sqrt])

    X_trainval_all = trainval_proc.drop(columns=TARGETS).select_dtypes(include="number")
    X_test_all = test_proc.drop(columns=TARGETS).select_dtypes(include="number")
    assert list(X_trainval_all.columns) == list(X_test_all.columns)

    scaler_X_tv = MinMaxScaler()
    X_trainval_scaled = pd.DataFrame(
        scaler_X_tv.fit_transform(X_trainval_all),
        columns=X_trainval_all.columns,
        index=X_trainval_all.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler_X_tv.transform(X_test_all), columns=X_test_all.columns, index=X_test_all.index
    )

    y = {}
    scalers_y_tv = {}
    for target in TARGETS:
        name = target.lower()
        y_trainval_sqrt = np.sqrt(trainval_proc[[target]])
        y_test_sqrt = np.sqrt(test_proc[[target]])

        scaler_y_tv = MinMaxScaler()
        y[f"y_trainval_{name}"] = pd.DataFrame(
            scaler_y_tv.fit_transform(y_trainval_sqrt), columns=[target], index=y_trainval_sqrt.index
        )
        y[f"y_test_{name}_tv"] = pd.DataFrame(
            scaler_y_tv.transform(y_test_sqrt), columns=[target], index=y_test_sqrt.index
        )
        scalers_y_tv[name] = scaler_y_tv

    X = {}
    for src in SOURCES:
        X[f"X_trainval_{src}"] = X_trainval_scaled.copy()
        X[f"X_test_{src}_tv"] = X_test_scaled.copy()

    return X, y, scaler_X_tv, scalers_y_tv


def drop_zero_variance_all_x(X, targets=TARGETS):
    """Notebook 2.7.1 - verbatim port of `drop_zero_variance_all_x`."""
    dropped = {}
    for target in targets:
        name = target.lower()
        X_train = X[f"X_train_{name}"]
        X_test = X[f"X_test_{name}"]

        numeric_X_train = X_train.select_dtypes(include="number")
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(numeric_X_train)
        kept_columns = numeric_X_train.columns[selector.get_support()]

        X[f"X_train_{name}"] = X_train[kept_columns]
        X[f"X_test_{name}"] = X_test[kept_columns]
        dropped[target] = X_train.shape[1] - len(kept_columns)
    return dropped


def mutual_info_scores(X_train, y_train, top_k=None, threshold=None):
    """Notebook 2.7 helper - verbatim port."""
    X_num = X_train.select_dtypes(include=["number"]).copy()

    if isinstance(y_train, pd.DataFrame):
        y = y_train.iloc[:, 0]
    elif isinstance(y_train, pd.Series):
        y = y_train
    else:
        y = np.asarray(y_train).reshape(-1)

    if isinstance(X_num, pd.DataFrame) and isinstance(y, pd.Series):
        X_num, y = X_num.align(y, join="inner", axis=0)

    valid = ~(X_num.isna().any(axis=1) | pd.isna(y))
    X_num, y = X_num.loc[valid], y.loc[valid]

    discrete_mask = (X_num.nunique() <= 10).values
    mi = mutual_info_regression(X_num, y, discrete_features=discrete_mask, random_state=0)

    scores = (
        pd.DataFrame({"features": X_num.columns, "Mutual Info Score": mi})
        .sort_values("Mutual Info Score", ascending=False)
        .reset_index(drop=True)
    )
    if threshold is not None:
        scores = scores[scores["Mutual Info Score"] > threshold].reset_index(drop=True)
    if top_k is not None:
        scores = scores.head(top_k)
    return scores


def xgboost_feature_importance(X_train, y_train, importance_type="gain", rel_thresh=0.0):
    """Notebook 2.7 helper - verbatim port (xgboost.XGBRegressor), with a
    GradientBoostingRegressor fallback, clearly flagged, if xgboost is not
    installed in this environment (see module docstring)."""
    X_num = X_train.select_dtypes(include=["number", "bool"]).copy()

    if isinstance(y_train, pd.DataFrame):
        y = y_train.iloc[:, 0].to_numpy().ravel()
    elif isinstance(y_train, pd.Series):
        y = y_train.to_numpy().ravel()
    else:
        y = np.asarray(y_train).ravel()

    if _HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_num, y)
        booster = model.get_booster()
        raw_imp = booster.get_score(importance_type=importance_type)
        name_map = {f"f{i}": col for i, col in enumerate(X_num.columns)}
        raw_imp = {name_map.get(k, k): v for k, v in raw_imp.items()}
        imp_series = pd.Series(raw_imp, dtype="float64").reindex(X_num.columns, fill_value=0.0)
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_num, y)
        imp_series = pd.Series(model.feature_importances_, index=X_num.columns, dtype="float64")

    df = pd.DataFrame({"features": imp_series.index, "importance": imp_series.values})
    total = df["importance"].sum()
    df["relative_importance (%)"] = 0.0 if total == 0 else 100.0 * df["importance"] / total

    if rel_thresh > 0:
        df = df[df["relative_importance (%)"] > rel_thresh]
    return df.sort_values("relative_importance (%)", ascending=False).reset_index(drop=True)


def select_features_per_target(X, y):
    """2.7.2-2.7.5 - automated equivalent of the original notebook's
    by-eye feature pick (see module docstring)."""
    selected = {}
    importance_tables = {}
    for target in TARGETS:
        name = target.lower()
        X_train = X[f"X_train_{name}"]
        y_train = y[f"y_train_{name}"]

        imp_df = xgboost_feature_importance(
            X_train, y_train, importance_type="gain", rel_thresh=1.0
        )
        importance_tables[target] = imp_df
        features = imp_df["features"].tolist()
        selected[target] = features

        for key in (f"X_train_{name}", f"X_val_{name}", f"X_test_{name}"):
            X[key] = X[key][features]
        for key in (f"X_trainval_{name}", f"X_test_{name}_tv"):
            X[key] = X[key][features]
    return selected, importance_tables


def main():
    stats = {}
    train, val, test = load_daily_splits()
    cols_to_sqrt = sqrt_input_columns(train)
    stats["n_sqrt_cols"] = len(cols_to_sqrt)
    stats["xgboost_available"] = _HAS_XGBOOST

    X1, y1, scaler_X1, scalers_y1 = scale_round1(train, val, test, cols_to_sqrt)
    X2, y2, scaler_X2, scalers_y2 = scale_round2_trainval(train, val, test, cols_to_sqrt)

    X = {**X1, **X2}
    y = {**y1, **y2}

    dropped_zero_var = drop_zero_variance_all_x(X)
    stats["zero_variance_cols_dropped"] = dropped_zero_var

    # keep round-2 (trainval/test) X aligned to the same zero-variance
    # columns kept for round 1, per target
    for target in TARGETS:
        name = target.lower()
        kept_cols = X[f"X_train_{name}"].columns
        X[f"X_trainval_{name}"] = X[f"X_trainval_{name}"][kept_cols]
        X[f"X_test_{name}_tv"] = X[f"X_test_{name}_tv"][kept_cols]

    selected, importance_tables = select_features_per_target(X, y)
    stats["selected_features"] = selected
    stats["n_selected_features"] = {t: len(f) for t, f in selected.items()}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(cols_to_sqrt, SCALER_DIR / "sqrt_input_columns.pkl")
    joblib.dump(scaler_X1, SCALER_DIR / "scaler_X_round1.pkl")
    joblib.dump(scaler_X2, SCALER_DIR / "scaler_X_round2_trainval.pkl")
    for name, sc in scalers_y1.items():
        joblib.dump(sc, SCALER_DIR / f"scaler_y_{name}_round1.pkl")
    for name, sc in scalers_y2.items():
        joblib.dump(sc, SCALER_DIR / f"scaler_y_{name}_round2_trainval.pkl")

    shapes = {}
    for target in TARGETS:
        name = target.lower()
        X_train = X[f"X_train_{name}"]
        X_val = X[f"X_val_{name}"]
        X_test = X[f"X_test_{name}"]
        X_trainval = X[f"X_trainval_{name}"]
        X_test_tv = X[f"X_test_{name}_tv"]

        # X and y get combined by position here, not by an index-based
        # join or concat. A handful of calendar dates sit right on the
        # boundary between two splits (the day training ends and
        # validation begins, say), and because 03_feature_engineering.py's
        # daily resample plus lag/rolling step is reproduced faithfully
        # from the notebook, that date can end up appearing twice in a
        # split's index, once from each adjacent split's partial-day
        # aggregation (see that script's docstring). X and y come from the
        # same source frame in the same row order, so they're already
        # aligned position-for-position; joining by index instead would
        # match the duplicated date label many-to-many and quietly inflate
        # the row count.
        def _attach_target(X_df, y_df, target_name):
            out = X_df.copy()
            out[target_name] = y_df[target_name].to_numpy()
            return out

        train_out = _attach_target(X_train, y[f"y_train_{name}"], target)
        val_out = _attach_target(X_val, y[f"y_val_{name}"], target)
        test_out = _attach_target(X_test, y[f"y_test_{name}"], target)
        trainval_out = _attach_target(X_trainval, y[f"y_trainval_{name}"], target)
        test_tv_out = _attach_target(X_test_tv, y[f"y_test_{name}_tv"], target)

        train_out.to_csv(PROCESSED_DIR / f"{name}_train.csv")
        val_out.to_csv(PROCESSED_DIR / f"{name}_val.csv")
        test_out.to_csv(PROCESSED_DIR / f"{name}_test.csv")
        trainval_out.to_csv(PROCESSED_DIR / f"{name}_trainval.csv")
        test_tv_out.to_csv(PROCESSED_DIR / f"{name}_test_tv.csv")

        importance_tables[target].to_csv(PROCESSED_DIR / f"{name}_xgb_importance.csv", index=False)

        shapes[target] = {
            "train": train_out.shape,
            "val": val_out.shape,
            "test": test_out.shape,
            "trainval": trainval_out.shape,
            "test_tv": test_tv_out.shape,
        }
    stats["final_shapes"] = shapes

    print("=== 04_scale_and_select.py ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return X, y, stats


if __name__ == "__main__":
    main()
