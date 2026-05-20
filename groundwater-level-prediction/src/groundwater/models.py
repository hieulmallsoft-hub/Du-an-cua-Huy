from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_naive_last(y_true: np.ndarray, lag_1_values: np.ndarray) -> Dict[str, float]:
    if len(y_true) != len(lag_1_values):
        raise ValueError("y_true and lag_1_values must have the same length")
    return evaluate_regression(y_true=y_true, y_pred=lag_1_values)


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: Dict[str, float]


def train_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, ModelResult]:
    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
        ),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate_regression(y_true=y_test, y_pred=pred)
        results[name] = ModelResult(name=name, model=model, metrics=metrics)
    return results


def select_best(results: Dict[str, ModelResult]) -> ModelResult:
    return sorted(
        results.values(),
        key=lambda x: (x.metrics["rmse"], x.metrics["mae"], -x.metrics["r2"]),
    )[0]
