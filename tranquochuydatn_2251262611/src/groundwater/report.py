from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_report(
    metrics_path: Path,
    predictions_path: Path,
    backtest_metrics_path: Path | None = None,
    include_plots: bool = True,
) -> str:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(predictions_path)

    best = metrics["selected_model"]
    by_model = metrics["metrics_by_model"]

    mae = preds["y_true"].sub(preds["y_pred"]).abs().mean()
    rmse = ((preds["y_true"] - preds["y_pred"]) ** 2).mean() ** 0.5
    top_error = preds.assign(abs_err=(preds["y_true"] - preds["y_pred"]).abs()).sort_values("abs_err", ascending=False).head(10)

    lines = []
    lines.append("# Groundwater Forecasting Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Selected model: `{best}`")
    lines.append(f"- Holdout MAE: `{mae:.6f}`")
    lines.append(f"- Holdout RMSE: `{rmse:.6f}`")
    requested = metrics.get("requested_primary_model")
    if requested and requested != best:
        lines.append(f"- Requested model: `{requested}`; selected model chosen by lowest holdout RMSE.")
    lines.append("")
    if "naive_last_baseline" in by_model:
        lines.append("Baseline benchmark included: `naive_last_baseline`")
        if best == "naive_last_baseline":
            lines.append("The naive baseline outperformed the trained model candidates on holdout RMSE.")
        lines.append("")

    lines.append("## Metrics by model")
    lines.append("| Model | MAE | RMSE | MAPE | R2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in by_model.items():
        lines.append(f"| {name} | {m['mae']:.6f} | {m['rmse']:.6f} | {m['mape']:.6f} | {m['r2']:.6f} |")
    lines.append("")
    lines.append("## Top 10 largest absolute errors")
    lines.append("| date | y_true | y_pred | abs_err |")
    lines.append("|---|---:|---:|---:|")
    for _, row in top_error.iterrows():
        lines.append(
            f"| {row['date']} | {row['y_true']:.6f} | {row['y_pred']:.6f} | {row['abs_err']:.6f} |"
        )
    lines.append("")

    if backtest_metrics_path and backtest_metrics_path.exists():
        btm = pd.read_csv(backtest_metrics_path)
        lines.append("## Post-train rolling-origin evaluation (multi-horizon)")
        lines.append("| Model | Horizon | MAE | RMSE | MAPE | R2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in btm.sort_values(["horizon", "model"]).iterrows():
            lines.append(
                f"| {row['model']} | {int(row['horizon'])} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} | {row['r2']:.6f} |"
            )
        lines.append("")
        pivot_rmse = btm.pivot(index="horizon", columns="model", values="rmse")
        if {"model", "naive_last"}.issubset(set(pivot_rmse.columns)):
            worse_horizons = pivot_rmse[pivot_rmse["model"] > pivot_rmse["naive_last"]].index.tolist()
            if worse_horizons:
                horizon_text = ", ".join(str(int(h)) for h in worse_horizons)
                lines.append(
                    f"Backtest note: the selected model has higher RMSE than `naive_last` at horizon(s): {horizon_text}."
                )
                lines.append("")

        if include_plots:
            lines.append("## Backtest plots")
            lines.append("- `artifacts/backtest_rmse.png`")
            lines.append("- `artifacts/backtest_curve_last_origin.png`")
            lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    metrics_path = Path(args.metrics)
    predictions_path = Path(args.predictions)
    out_path = Path(args.out)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    backtest_path = Path(args.backtest_metrics) if args.backtest_metrics else None
    report = build_report(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        backtest_metrics_path=backtest_path,
        include_plots=not args.no_plot_refs,
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate markdown report from metrics and predictions.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics JSON")
    parser.add_argument("--predictions", type=str, required=True, help="Path to test predictions CSV")
    parser.add_argument("--backtest-metrics", type=str, default="", help="Optional backtest metrics CSV")
    parser.add_argument("--no-plot-refs", action="store_true", help="Disable plot references in report")
    parser.add_argument("--out", type=str, default="artifacts/report.md", help="Output markdown path")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
