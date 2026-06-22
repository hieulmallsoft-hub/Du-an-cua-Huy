from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import load_adjacent_metadata, load_series, parse_optional_columns
from .hybrid import HybridArtifact, forecast_hybrid, train_hybrid
from .lstm_model import LSTMArtifact, predict_lstm_steps, train_lstm
from .models import evaluate_regression
from .ts_models import VARFit, VECMFit, adf_pvalues, select_var_fit, select_vecm_fit, var_forecast, vecm_forecast


def split_by_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n_test = max(1, int(len(df) * test_size))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    if train.empty or test.empty:
        raise ValueError("Data split produced empty train or test set")
    return train, test


def _metric_payload(y_true: np.ndarray, y_pred: List[float]) -> Dict[str, float]:
    return evaluate_regression(y_true=y_true, y_pred=np.array(y_pred))


def _select_primary_model(
    requested: str,
    metrics_by_model: Dict[str, Dict[str, float]],
    predictions_by_model: Dict[str, List[float]],
) -> str:
    if requested != "auto":
        if requested not in predictions_by_model:
            raise ValueError(f"primary_model must be one of: {sorted(predictions_by_model.keys()) + ['auto']}")
        return requested

    return sorted(
        predictions_by_model.keys(),
        key=lambda name: (
            metrics_by_model[name]["rmse"],
            metrics_by_model[name]["mae"],
            -metrics_by_model[name]["r2"],
        ),
    )[0]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exogenous_cols = parse_optional_columns(args.feature_cols)

    raw_df = load_series(
        data_path=args.data,
        date_col=args.date_col,
        target_col=args.target_col,
        feature_cols=exogenous_cols,
    )

    derived_cols: List[str] = []
    if not exogenous_cols:
        diff_col = f"{args.target_col}_diff"
        raw_df[diff_col] = raw_df[args.target_col].diff().fillna(0.0)
        exogenous_cols = [diff_col]
        derived_cols.append(diff_col)

    endog_cols = [args.target_col] + [col for col in exogenous_cols if col != args.target_col]
    endog_df = raw_df[endog_cols].astype(float)

    train_df, test_df = split_by_time(raw_df, test_size=args.test_size)
    train_endog = train_df[endog_cols].astype(float)
    test_endog = test_df[endog_cols].astype(float)

    y_test = test_endog[args.target_col].to_numpy()
    target_idx = endog_cols.index(args.target_col)

    diagnostics = {
        "adf_pvalues": adf_pvalues(train_endog),
        "derived_exogenous_cols": derived_cols,
    }

    var_fit: VARFit | None = None
    vecm_fit: VECMFit | None = None
    lstm_artifact: LSTMArtifact | None = None
    hybrid_artifact: HybridArtifact | None = None

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    predictions_by_model: Dict[str, List[float]] = {}

    var_fit = select_var_fit(train_endog, maxlags=args.maxlags, ic="aic")
    var_fc = var_forecast(var_fit, train_endog.to_numpy(), steps=len(test_endog))
    var_pred = var_fc[:, target_idx].tolist()
    metrics_by_model["var"] = _metric_payload(y_test, var_pred)
    predictions_by_model["var"] = var_pred

    if len(endog_cols) > 1:
        try:
            vecm_fit = select_vecm_fit(train_endog, maxlags=args.maxlags, deterministic="co")
            vecm_fc = vecm_forecast(vecm_fit, steps=len(test_endog))
            vecm_pred = vecm_fc[:, target_idx].tolist()
            metrics_by_model["vecm"] = _metric_payload(y_test, vecm_pred)
            predictions_by_model["vecm"] = vecm_pred
        except Exception as exc:
            diagnostics["vecm_error"] = str(exc)

    lstm_artifact, _ = train_lstm(
        series=train_endog[args.target_col].to_numpy(),
        seq_len=args.lstm_seq_len,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        epochs=args.lstm_epochs,
        lr=args.lstm_lr,
        random_state=args.random_state,
    )
    lstm_pred = predict_lstm_steps(lstm_artifact, train_endog[args.target_col].to_numpy(), steps=len(test_endog))
    metrics_by_model["lstm"] = _metric_payload(y_test, lstm_pred)
    predictions_by_model["lstm"] = lstm_pred

    hybrid_base = args.hybrid_base
    if hybrid_base == "vecm" and len(endog_cols) < 2:
        hybrid_base = "var"
    hybrid_artifact = train_hybrid(
        endog_df=train_endog,
        target_col=args.target_col,
        maxlags=args.maxlags,
        base_type=hybrid_base,
        seq_len=args.lstm_seq_len,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        epochs=args.lstm_epochs,
        lr=args.lstm_lr,
        random_state=args.random_state,
    )
    hybrid_pred = forecast_hybrid(
        hybrid_artifact,
        history_matrix=train_endog.to_numpy(),
        steps=len(test_endog),
        target_col=args.target_col,
    )
    metrics_by_model["hybrid"] = _metric_payload(y_test, hybrid_pred)
    predictions_by_model["hybrid"] = hybrid_pred

    naive_pred = [float(train_endog[args.target_col].iloc[-1])] * len(test_endog)
    metrics_by_model["naive_last_baseline"] = _metric_payload(y_test, naive_pred)
    predictions_by_model["naive_last_baseline"] = naive_pred

    selected_model = _select_primary_model(
        requested=args.primary_model,
        metrics_by_model=metrics_by_model,
        predictions_by_model=predictions_by_model,
    )

    primary_pred = predictions_by_model[selected_model]
    pred_frame = pd.DataFrame(
        {
            "date": test_df[args.date_col].astype(str).tolist(),
            "y_true": y_test,
            "y_pred": primary_pred,
        }
    )
    pred_frame.to_csv(out_dir / "test_predictions.csv", index=False)
    pred_frame.to_csv(out_dir / "test_predictions_tuned.csv", index=False)

    history_len = max(args.lstm_seq_len, args.maxlags + 5, 30)
    default_target_history = train_endog[args.target_col].tail(history_len).tolist()
    default_endog_history = train_endog.tail(history_len).values.tolist()

    artifact = {
        "artifact_version": "thesis_v1",
        "model_type": selected_model,
        "model_name": selected_model,
        "date_col": args.date_col,
        "target_col": args.target_col,
        "endogenous_cols": endog_cols,
        "exogenous_cols": exogenous_cols,
        "default_target_history": default_target_history,
        "default_endog_history": default_endog_history,
        "metrics_by_model": metrics_by_model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_data": str(args.data),
        "split": {
            "rows_total": int(len(raw_df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "test_size": float(args.test_size),
        },
        "random_state": args.random_state,
        "var_fit": var_fit,
        "vecm_fit": vecm_fit,
        "lstm_artifact": lstm_artifact,
        "hybrid_artifact": hybrid_artifact,
        "diagnostics": diagnostics,
    }

    with (out_dir / args.artifact_name).open("wb") as f:
        pickle.dump(artifact, f)

    metrics_payload = {
        "selected_model": selected_model,
        "requested_primary_model": args.primary_model,
        "source_data": str(args.data),
        "data_metadata": load_adjacent_metadata(args.data),
        "metrics_by_model": metrics_by_model,
        "config": {
            "test_size": args.test_size,
            "feature_cols": exogenous_cols,
            "maxlags": args.maxlags,
            "lstm_seq_len": args.lstm_seq_len,
            "lstm_hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "lstm_epochs": args.lstm_epochs,
            "lstm_lr": args.lstm_lr,
            "hybrid_base": hybrid_base,
            "random_state": args.random_state,
        },
        "diagnostics": diagnostics,
        "split": {
            "rows_total": int(len(raw_df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
        },
    }
    (out_dir / "tuned_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Thesis training complete")
    print(f"- Selected model: {selected_model}")
    print(f"- Artifact: {out_dir / args.artifact_name}")
    print(f"- Metrics: {out_dir / 'tuned_metrics.json'}")
    print(f"- Test predictions: {out_dir / 'test_predictions.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train thesis models: VAR/VECM/LSTM + hybrid.")
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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test ratio for time split")
    parser.add_argument("--maxlags", type=int, default=12, help="Max lag order for VAR/VECM")
    parser.add_argument("--primary-model", type=str, default="auto", help="auto | var | vecm | lstm | hybrid | naive_last_baseline")
    parser.add_argument("--hybrid-base", type=str, default="var", help="Base linear model for hybrid: var | vecm")
    parser.add_argument("--lstm-seq-len", type=int, default=14, help="Sequence length for LSTM")
    parser.add_argument("--lstm-hidden-size", type=int, default=32, help="Hidden size for LSTM")
    parser.add_argument("--lstm-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--lstm-epochs", type=int, default=50, help="Training epochs for LSTM")
    parser.add_argument("--lstm-lr", type=float, default=1e-3, help="Learning rate for LSTM")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible training")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
