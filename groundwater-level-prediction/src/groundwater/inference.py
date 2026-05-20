from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .features import build_feature_vector_from_history
from .hybrid import HybridArtifact, forecast_hybrid
from .lstm_model import LSTMArtifact, predict_lstm_steps
from .ts_models import VARFit, VECMFit, var_forecast, vecm_forecast


class GroundwaterService:
    def __init__(self, artifact: Dict) -> None:
        self.artifact_version = artifact.get("artifact_version", "baseline")
        self.model_name = artifact.get("model_name")
        self.date_col = artifact.get("date_col")
        self.target_col = artifact.get("target_col")
        self.source_data = artifact.get("source_data")
        self.training_metrics = artifact.get("metrics_by_model", {})
        self.trained_at = artifact.get("trained_at")
        self.split = artifact.get("split", {})

        if self.artifact_version == "thesis_v1":
            self.model_type = artifact.get("model_type")
            self.endogenous_cols = artifact.get("endogenous_cols", [])
            self.exogenous_cols = artifact.get("exogenous_cols", [])
            self.default_history = artifact.get("default_target_history", [])
            self.default_endog_history = artifact.get("default_endog_history", [])
            self.lags = artifact.get("lags", [])
            self.rolling_windows = artifact.get("rolling_windows", [])
            self.horizon = artifact.get("horizon", 1)
            self.var_fit: VARFit | None = artifact.get("var_fit")
            self.vecm_fit: VECMFit | None = artifact.get("vecm_fit")
            self.lstm_artifact: LSTMArtifact | None = artifact.get("lstm_artifact")
            self.hybrid_artifact: HybridArtifact | None = artifact.get("hybrid_artifact")
        else:
            self.model = artifact["model"]
            self.date_col = artifact["date_col"]
            self.lags = artifact["lags"]
            self.rolling_windows = artifact["rolling_windows"]
            self.horizon = artifact["horizon"]
            self.feature_columns = artifact["feature_columns"]
            self.default_history = artifact.get("default_history", [])
            self.exogenous_cols = artifact.get("exogenous_cols", [])
            self.exogenous_defaults = artifact.get("exogenous_defaults", {})

    def _vector_from_history(
        self,
        history_levels: Sequence[float],
        exogenous_values: Dict[str, float] | None = None,
    ) -> np.ndarray:
        merged_exogenous = dict(self.exogenous_defaults)
        if exogenous_values:
            merged_exogenous.update({str(k): float(v) for k, v in exogenous_values.items()})

        feats = build_feature_vector_from_history(
            history_levels=history_levels,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            exogenous_values=merged_exogenous,
            feature_columns=self.feature_columns,
        )
        vector = np.array([feats[col] for col in self.feature_columns], dtype=float).reshape(1, -1)
        return vector

    def _build_endog_history(
        self,
        history_levels: Sequence[float] | None,
        exogenous_values: Dict[str, float] | None,
    ) -> np.ndarray:
        base = np.array(self.default_endog_history, dtype=float)
        if base.ndim != 2 or base.shape[1] != len(self.endogenous_cols):
            raise ValueError("default_endog_history is not compatible with endogenous_cols")

        target_idx = self.endogenous_cols.index(self.target_col)
        if history_levels:
            hist = np.array(history_levels, dtype=float)
            use = hist[-base.shape[0] :]
            base[-len(use) :, target_idx] = use

            # Recompute derived diff column if present
            diff_col = f"{self.target_col}_diff"
            if diff_col in self.endogenous_cols:
                idx = self.endogenous_cols.index(diff_col)
                diff_vals = np.diff(base[:, target_idx], prepend=base[0, target_idx])
                base[:, idx] = diff_vals

        if exogenous_values:
            for key, val in exogenous_values.items():
                if key in self.endogenous_cols and key != self.target_col:
                    idx = self.endogenous_cols.index(key)
                    base[-1, idx] = float(val)
        return base

    def predict_next(
        self,
        history_levels: Sequence[float] | None = None,
        exogenous_values: Dict[str, float] | None = None,
    ) -> float:
        history = history_levels if history_levels is not None else self.default_history
        if not history:
            raise ValueError("No history provided and no default_history in artifact.")

        if self.artifact_version != "thesis_v1":
            vector = self._vector_from_history(history, exogenous_values=exogenous_values)
            return float(self.model.predict(vector)[0])

        if self.model_type == "lstm" and self.lstm_artifact is not None:
            return float(predict_lstm_steps(self.lstm_artifact, np.array(history), steps=1)[0])
        if self.model_type == "var" and self.var_fit is not None:
            hist = self._build_endog_history(history, exogenous_values)
            target_idx = self.endogenous_cols.index(self.target_col)
            return float(var_forecast(self.var_fit, hist, steps=1)[0, target_idx])
        if self.model_type == "vecm" and self.vecm_fit is not None:
            target_idx = self.endogenous_cols.index(self.target_col)
            return float(vecm_forecast(self.vecm_fit, steps=1)[0, target_idx])
        if self.model_type == "hybrid" and self.hybrid_artifact is not None:
            hist = self._build_endog_history(history, exogenous_values)
            return float(forecast_hybrid(self.hybrid_artifact, hist, steps=1, target_col=self.target_col)[0])
        if self.model_type == "naive_last_baseline":
            return float(history[-1])

        raise ValueError("Unsupported model_type for predict_next")

    def forecast(
        self,
        history_levels: Sequence[float] | None = None,
        steps: int = 7,
        exogenous_values: Dict[str, float] | None = None,
        exogenous_sequence: List[Dict[str, float]] | None = None,
    ) -> List[float]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        rolling_history = list(history_levels if history_levels is not None else self.default_history)
        if not rolling_history:
            raise ValueError("No history provided and no default_history in artifact.")
        if exogenous_sequence is not None and len(exogenous_sequence) != steps:
            raise ValueError("exogenous_sequence must have the same length as steps")

        if self.artifact_version != "thesis_v1":
            preds: List[float] = []
            for i in range(steps):
                step_exogenous = exogenous_sequence[i] if exogenous_sequence is not None else exogenous_values
                pred = self.predict_next(rolling_history, exogenous_values=step_exogenous)
                preds.append(pred)
                rolling_history.append(pred)
            return preds

        if self.model_type == "lstm" and self.lstm_artifact is not None:
            return predict_lstm_steps(self.lstm_artifact, np.array(rolling_history), steps=steps)
        if self.model_type == "var" and self.var_fit is not None:
            hist = self._build_endog_history(rolling_history, exogenous_values)
            target_idx = self.endogenous_cols.index(self.target_col)
            fc = var_forecast(self.var_fit, hist, steps=steps)
            return fc[:, target_idx].tolist()
        if self.model_type == "vecm" and self.vecm_fit is not None:
            target_idx = self.endogenous_cols.index(self.target_col)
            return vecm_forecast(self.vecm_fit, steps=steps)[:, target_idx].tolist()
        if self.model_type == "hybrid" and self.hybrid_artifact is not None:
            hist = self._build_endog_history(rolling_history, exogenous_values)
            return forecast_hybrid(self.hybrid_artifact, hist, steps=steps, target_col=self.target_col)
        if self.model_type == "naive_last_baseline":
            return [float(rolling_history[-1])] * steps

        raise ValueError("Unsupported model_type for forecast")


def load_service(artifact_path: str | Path) -> GroundwaterService:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("rb") as f:
        artifact = pickle.load(f)
    return GroundwaterService(artifact)
