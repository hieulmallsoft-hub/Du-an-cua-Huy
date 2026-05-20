from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import load_series, parse_optional_columns
from .features import build_supervised_frame, parse_int_list
from .models import select_best, train_candidates


def split_by_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n_test = max(1, int(len(df) * test_size))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    if train.empty or test.empty:
        raise ValueError("Data split produced empty train or test set")
    return train, test


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

    results = train_candidates(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=42,
    )
    best = select_best(results)

    metrics_by_model = {name: res.metrics for name, res in results.items()}
    preds = best.model.predict(X_test)

    required_history = max(max(lags), max(rolling_windows))
    default_history = raw_df[args.target_col].tail(required_history).tolist()
    exogenous_defaults = {}
    if exogenous_cols:
        latest = raw_df.iloc[-1]
        exogenous_defaults = {col: float(latest[col]) for col in exogenous_cols}

    artifact = {
        "model": best.model,
        "model_name": best.name,
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
    }

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(artifact, f)

    metrics_payload = {
        "selected_model": best.name,
        "metrics_by_model": metrics_by_model,
        "config": {
            "lags": lags,
            "rolling_windows": rolling_windows,
            "horizon": args.horizon,
            "test_size": args.test_size,
            "feature_cols": exogenous_cols,
        },
        "split": {
            "rows_total": int(len(supervised)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    prediction_frame = pd.DataFrame(
        {
            "date": test_df[args.date_col].astype(str).tolist(),
            "y_true": y_test,
            "y_pred": preds,
        }
    )
    prediction_frame.to_csv(out_dir / "test_predictions.csv", index=False)

    print("Training complete")
    print(f"- Selected model: {best.name}")
    print(f"- Artifact: {out_dir / 'model.pkl'}")
    print(f"- Metrics: {out_dir / 'metrics.json'}")
    print(f"- Test predictions: {out_dir / 'test_predictions.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train groundwater level forecasting baselines.")
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
    parser.add_argument("--lags", type=str, default="1,2,3,7,14,30", help="Comma-separated lag list")
    parser.add_argument("--rolling-windows", type=str, default="7,14,30", help="Comma-separated rolling windows")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test ratio for time split")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
