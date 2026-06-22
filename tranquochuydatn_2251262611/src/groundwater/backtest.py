from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import load_series
from .inference import load_service
from .models import evaluate_regression


def parse_horizons(value: str) -> List[int]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    horizons = sorted({int(x) for x in parts if int(x) > 0})
    if not horizons:
        raise ValueError("horizons cannot be empty")
    return horizons


def metric_frame(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    result = evaluate_regression(y_true=y_true, y_pred=y_pred)
    return {
        "mae": float(result["mae"]),
        "rmse": float(result["rmse"]),
        "mape": float(result["mape"]),
        "r2": float(result["r2"]),
    }


def naive_last_forecast(history: List[float], steps: int) -> List[float]:
    if not history:
        raise ValueError("history cannot be empty")
    return [float(history[-1])] * steps


def build_origin_indices(
    n: int,
    max_h: int,
    min_train_size: int,
    stride: int,
    min_origin_index: int | None = None,
) -> List[int]:
    start = max(min_train_size - 1, 0)
    if min_origin_index is not None:
        start = max(start, min_origin_index)
    end = n - max_h - 1
    if end <= start:
        return []
    return list(range(start, end + 1, max(1, stride)))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = load_service(args.artifact)
    df = load_series(args.data, date_col=args.date_col, target_col=args.target_col, feature_cols=svc.exogenous_cols)
    values = df[args.target_col].astype(float).to_numpy()
    dates = df[args.date_col].astype(str).to_numpy()
    endogenous_cols = getattr(svc, "endogenous_cols", [])
    endog_values = df[endogenous_cols].astype(float).to_numpy() if endogenous_cols else None

    horizons = parse_horizons(args.horizons)
    max_h = max(horizons)
    split = getattr(svc, "split", {}) or {}
    artifact_rows_train = split.get("rows_train")
    min_origin_index = None
    evaluation_mode = "rolling_origin_loaded_model"
    if artifact_rows_train is not None and not args.allow_in_sample_origins:
        min_origin_index = max(0, int(artifact_rows_train) - 1)
        evaluation_mode = "post_train_rolling_origin_loaded_model"

    origin_idx = build_origin_indices(
        n=len(df),
        max_h=max_h,
        min_train_size=args.min_train_size,
        stride=args.stride,
        min_origin_index=min_origin_index,
    )
    if not origin_idx:
        raise RuntimeError("Not enough rows for backtesting with current configuration.")

    rows = []
    for origin in origin_idx:
        history = values[: origin + 1].tolist()
        endogenous_history = endog_values[: origin + 1] if endog_values is not None else None
        model_forecast = svc.forecast(
            history_levels=history,
            steps=max_h,
            endogenous_history=endogenous_history,
        )
        naive_forecast = naive_last_forecast(history, steps=max_h)
        for h in horizons:
            true_idx = origin + h
            y_true = float(values[true_idx])
            rows.append(
                {
                    "origin_index": int(origin),
                    "origin_date": str(dates[origin]),
                    "target_date": str(dates[true_idx]),
                    "horizon": int(h),
                    "y_true": y_true,
                    "y_pred_model": float(model_forecast[h - 1]),
                    "y_pred_naive": float(naive_forecast[h - 1]),
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_df["abs_err_model"] = np.abs(pred_df["y_true"] - pred_df["y_pred_model"])
    pred_df["abs_err_naive"] = np.abs(pred_df["y_true"] - pred_df["y_pred_naive"])

    metrics_rows = []
    summary_payload: Dict[str, Dict[str, Dict[str, float]]] = {"model": {}, "naive_last": {}}
    for h in horizons:
        h_df = pred_df[pred_df["horizon"] == h]
        model_m = metric_frame(h_df["y_true"].to_numpy(), h_df["y_pred_model"].to_numpy())
        naive_m = metric_frame(h_df["y_true"].to_numpy(), h_df["y_pred_naive"].to_numpy())
        summary_payload["model"][str(h)] = model_m
        summary_payload["naive_last"][str(h)] = naive_m

        for name, m in [("model", model_m), ("naive_last", naive_m)]:
            metrics_rows.append(
                {
                    "model": name,
                    "horizon": h,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "r2": m["r2"],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "backtest_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "backtest_predictions.csv", index=False)
    (out_dir / "backtest_summary.json").write_text(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "data": str(args.data),
                "horizons": horizons,
                "origins": len(origin_idx),
                "min_train_size": args.min_train_size,
                "stride": args.stride,
                "evaluation_mode": evaluation_mode,
                "first_origin_index": int(origin_idx[0]),
                "last_origin_index": int(origin_idx[-1]),
                "artifact_rows_train": int(artifact_rows_train) if artifact_rows_train is not None else None,
                "allow_in_sample_origins": bool(args.allow_in_sample_origins),
                "metrics": summary_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plot charts when matplotlib is available.
    if args.make_plots:
        try:
            import matplotlib.pyplot as plt

            pivot_rmse = metrics_df.pivot(index="horizon", columns="model", values="rmse")
            ax = pivot_rmse.plot(kind="bar", figsize=(8, 4), title="RMSE by Horizon (Backtest)")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("RMSE")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "backtest_rmse.png", dpi=150)
            plt.close()

            # Example: last backtest origin forecast curve
            last_origin = pred_df["origin_index"].max()
            last_df = pred_df[pred_df["origin_index"] == last_origin].sort_values("horizon")
            plt.figure(figsize=(8, 4))
            plt.plot(last_df["horizon"], last_df["y_true"], marker="o", label="Actual")
            plt.plot(last_df["horizon"], last_df["y_pred_model"], marker="o", label="Model")
            plt.plot(last_df["horizon"], last_df["y_pred_naive"], marker="o", label="Naive last")
            plt.title("Forecast Curve at Last Backtest Origin")
            plt.xlabel("Horizon")
            plt.ylabel(args.target_col)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "backtest_curve_last_origin.png", dpi=150)
            plt.close()
        except Exception:
            pass

    print("Backtest complete")
    print(f"- Predictions: {out_dir / 'backtest_predictions.csv'}")
    print(f"- Metrics: {out_dir / 'backtest_metrics.csv'}")
    print(f"- Summary: {out_dir / 'backtest_summary.json'}")
    if args.make_plots:
        print(f"- Plots: {out_dir / 'backtest_rmse.png'}, {out_dir / 'backtest_curve_last_origin.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtesting for groundwater forecasting.")
    parser.add_argument("--artifact", type=str, default="artifacts/model.pkl", help="Path to trained artifact")
    parser.add_argument("--data", type=str, required=True, help="Path to source data CSV")
    parser.add_argument("--date-col", type=str, default="date", help="Date column name")
    parser.add_argument("--target-col", type=str, default="groundwater_level", help="Target column name")
    parser.add_argument("--horizons", type=str, default="1,3,7,14", help="Comma-separated forecast horizons")
    parser.add_argument("--min-train-size", type=int, default=365, help="Minimum history length before first origin")
    parser.add_argument("--stride", type=int, default=7, help="Origin stride size")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--make-plots", action="store_true", help="Generate png plots")
    parser.add_argument(
        "--allow-in-sample-origins",
        action="store_true",
        help="Allow origins before the artifact training window ends. This is not leakage-safe for a pre-trained artifact.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
