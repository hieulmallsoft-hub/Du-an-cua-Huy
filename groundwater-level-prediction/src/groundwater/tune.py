from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from .data import load_series, parse_optional_columns
from .features import build_supervised_frame, parse_int_list
from .models import evaluate_naive_last, evaluate_regression


def split_by_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n_test = max(1, int(len(df) * test_size))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    if train.empty or test.empty:
        raise ValueError("Data split produced empty train or test set")
    return train, test


def build_param_list(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for item in product(*values):
        combos.append(dict(zip(keys, item)))
    return combos


def cv_rmse(
    model_name: str,
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores: List[float] = []

    for train_idx, valid_idx in tscv.split(X_train):
        X_tr, X_va = X_train[train_idx], X_train[valid_idx]
        y_tr, y_va = y_train[train_idx], y_train[valid_idx]

        if model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=42, n_jobs=1, **params)
        elif model_name == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
        fold_scores.append(rmse)

    return float(np.mean(fold_scores))


def fit_model(model_name: str, params: Dict, X_train: np.ndarray, y_train: np.ndarray):
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=1, **params)
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42, **params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    return model


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lags = parse_int_list(args.lags)
    rolling_windows = parse_int_list(args.rolling_windows)
    exogenous_cols = parse_optional_columns(args.feature_cols)

    raw_df = load_series(
        data_path=args.data,
        date_col=args.date_col,
        target_col=args.target_col,
        feature_cols=exogenous_cols,
    )
    supervised, feature_columns = build_supervised_frame(
        df=raw_df,
        date_col=args.date_col,
        target_col=args.target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        exogenous_cols=exogenous_cols,
        horizon=args.horizon,
    )
    train_df, test_df = split_by_time(supervised, test_size=args.test_size)

    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df["target_future"].to_numpy()
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df["target_future"].to_numpy()

    param_space = {
        "linear_regression": [{}],
        "random_forest": build_param_list(
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 4],
            }
        ),
        "gradient_boosting": build_param_list(
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            }
        ),
    }

    tuning_rows = []
    tuned_models = {}
    metrics_by_model = {}

    for model_name, candidates in param_space.items():
        best_cv = float("inf")
        best_params = {}
        for params in candidates:
            score = cv_rmse(
                model_name=model_name,
                params=params,
                X_train=X_train,
                y_train=y_train,
                n_splits=args.cv_splits,
            )
            tuning_rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, sort_keys=True),
                    "cv_rmse": round(score, 6),
                }
            )
            if score < best_cv:
                best_cv = score
                best_params = params

        model = fit_model(model_name, best_params, X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate_regression(y_true=y_test, y_pred=pred)
        metrics["cv_rmse"] = best_cv
        metrics["best_params"] = best_params
        tuned_models[model_name] = model
        metrics_by_model[model_name] = metrics

    # Benchmark baseline: predict next value = last observed value (lag_1)
    if "lag_1" in feature_columns:
        lag_1_pred = test_df["lag_1"].to_numpy()
        naive_metrics = evaluate_naive_last(y_true=y_test, lag_1_values=lag_1_pred)
        naive_metrics["cv_rmse"] = None
        naive_metrics["best_params"] = {}
        metrics_by_model["naive_last_baseline"] = naive_metrics

    # Select deployment model only from trainable ML candidates
    best_model_name = sorted(
        tuned_models.keys(),
        key=lambda name: (
            metrics_by_model[name]["rmse"],
            metrics_by_model[name]["mae"],
            -metrics_by_model[name]["r2"],
        ),
    )[0]
    best_model = tuned_models[best_model_name]

    required_history = max(max(lags), max(rolling_windows))
    default_history = raw_df[args.target_col].tail(required_history).tolist()
    exogenous_defaults = {}
    if exogenous_cols:
        latest = raw_df.iloc[-1]
        exogenous_defaults = {col: float(latest[col]) for col in exogenous_cols}

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "date_col": args.date_col,
        "target_col": args.target_col,
        "lags": lags,
        "rolling_windows": rolling_windows,
        "horizon": args.horizon,
        "feature_columns": feature_columns,
        "metrics_by_model": metrics_by_model,
        "default_history": default_history,
        "exogenous_cols": exogenous_cols,
        "exogenous_defaults": exogenous_defaults,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_data": str(args.data),
        "tuned": True,
    }
    with (out_dir / args.artifact_name).open("wb") as f:
        pickle.dump(artifact, f)

    tuning_df = pd.DataFrame(tuning_rows).sort_values(["model", "cv_rmse"]).reset_index(drop=True)
    tuning_df.to_csv(out_dir / "tuning_results.csv", index=False)

    test_pred = best_model.predict(X_test)
    pred_frame = pd.DataFrame(
        {
            "date": test_df[args.date_col].astype(str).tolist(),
            "y_true": y_test,
            "y_pred": test_pred,
            "abs_error": np.abs(y_test - test_pred),
        }
    )
    pred_frame["ape_percent"] = np.where(
        np.abs(pred_frame["y_true"]) < 1e-6,
        0.0,
        (pred_frame["abs_error"] / np.abs(pred_frame["y_true"])) * 100.0,
    )
    pred_frame.to_csv(out_dir / "test_predictions_tuned.csv", index=False)

    metrics_payload = {
        "selected_model": best_model_name,
        "metrics_by_model": metrics_by_model,
        "config": {
            "lags": lags,
            "rolling_windows": rolling_windows,
            "horizon": args.horizon,
            "test_size": args.test_size,
            "feature_cols": exogenous_cols,
            "cv_splits": args.cv_splits,
        },
        "split": {
            "rows_total": int(len(supervised)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
        },
    }
    (out_dir / "tuned_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Tuning complete")
    print(f"- Selected tuned model: {best_model_name}")
    print(f"- Artifact: {out_dir / args.artifact_name}")
    print(f"- Metrics: {out_dir / 'tuned_metrics.json'}")
    print(f"- Tuning table: {out_dir / 'tuning_results.csv'}")
    print(f"- Test predictions: {out_dir / 'test_predictions_tuned.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune groundwater forecasting models with time-series CV.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with date + groundwater level")
    parser.add_argument("--date-col", type=str, default="date", help="Date column name")
    parser.add_argument("--target-col", type=str, default="groundwater_level", help="Target column name")
    parser.add_argument(
        "--feature-cols",
        type=str,
        default="",
        help="Optional comma-separated exogenous numeric feature columns",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--artifact-name", type=str, default="model.pkl", help="Output model artifact name")
    parser.add_argument("--lags", type=str, default="1,2,3,7,14,30", help="Comma-separated lag list")
    parser.add_argument("--rolling-windows", type=str, default="7,14,30", help="Comma-separated rolling windows")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test ratio for time split")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of TimeSeries CV splits")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
