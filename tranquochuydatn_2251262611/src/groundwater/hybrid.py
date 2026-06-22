from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .lstm_model import LSTMArtifact, predict_lstm_steps, train_lstm
from .ts_models import VARFit, VECMFit, select_var_fit, select_vecm_fit, var_forecast, vecm_forecast


@dataclass
class HybridArtifact:
    base_type: str
    var_fit: VARFit | None
    vecm_fit: VECMFit | None
    residual_lstm: LSTMArtifact
    endog_columns: List[str]
    base_history: List[List[float]]


def _fit_base(
    endog_df: pd.DataFrame,
    base_type: str,
    maxlags: int,
) -> tuple[VARFit | None, VECMFit | None, np.ndarray]:
    if base_type == "var":
        var_fit = select_var_fit(endog_df, maxlags=maxlags, ic="aic")
        fitted = var_fit.model.fittedvalues
        return var_fit, None, fitted
    if base_type == "vecm":
        vecm_fit = select_vecm_fit(endog_df, maxlags=maxlags, deterministic="co")
        fitted = vecm_fit.model.fittedvalues
        return None, vecm_fit, fitted
    raise ValueError("base_type must be 'var' or 'vecm'")


def train_hybrid(
    endog_df: pd.DataFrame,
    target_col: str,
    maxlags: int = 12,
    base_type: str = "var",
    seq_len: int = 14,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    random_state: int = 42,
) -> HybridArtifact:
    var_fit, vecm_fit, fitted = _fit_base(endog_df, base_type=base_type, maxlags=maxlags)
    fitted_target = fitted[target_col].to_numpy()
    actual = endog_df[target_col].iloc[-len(fitted_target) :].to_numpy()
    residuals = actual - fitted_target

    lstm_artifact, _ = train_lstm(
        series=residuals,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        random_state=random_state,
    )

    base_history = endog_df.tail(maxlags + 5).values.tolist()
    return HybridArtifact(
        base_type=base_type,
        var_fit=var_fit,
        vecm_fit=vecm_fit,
        residual_lstm=lstm_artifact,
        endog_columns=list(endog_df.columns),
        base_history=base_history,
    )


def forecast_hybrid(
    artifact: HybridArtifact,
    history_matrix: np.ndarray,
    steps: int,
    target_col: str,
) -> List[float]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    target_idx = artifact.endog_columns.index(target_col)

    if artifact.base_type == "var" and artifact.var_fit is not None:
        base_fc = var_forecast(artifact.var_fit, history_matrix, steps=steps)
    elif artifact.base_type == "vecm" and artifact.vecm_fit is not None:
        base_fc = vecm_forecast(artifact.vecm_fit, steps=steps, history=history_matrix)
    else:
        raise ValueError("Hybrid base model missing")

    base_target = base_fc[:, target_idx]
    residual_history = np.array([], dtype=float)
    if artifact.var_fit is not None:
        fitted = artifact.var_fit.model.fittedvalues
        fitted_target = fitted[target_col].to_numpy()
        take = min(len(fitted_target), history_matrix.shape[0])
        actual = history_matrix[-take:, target_idx]
        residual_history = actual - fitted_target[-take:]
    elif artifact.vecm_fit is not None:
        fitted = artifact.vecm_fit.model.fittedvalues
        fitted_target = fitted[target_col].to_numpy()
        take = min(len(fitted_target), history_matrix.shape[0])
        actual = history_matrix[-take:, target_idx]
        residual_history = actual - fitted_target[-take:]

    if residual_history.size == 0:
        residual_history = history_matrix[:, target_idx] * 0.0

    residual_fc = predict_lstm_steps(artifact.residual_lstm, residual_history, steps=steps)
    return (base_target + np.array(residual_fc)).tolist()
