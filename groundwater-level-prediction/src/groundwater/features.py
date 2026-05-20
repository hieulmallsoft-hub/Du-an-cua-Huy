from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_int_list(value: str) -> List[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    parsed = sorted({int(x) for x in items if int(x) > 0})
    if not parsed:
        raise ValueError("List of integers cannot be empty")
    return parsed


def build_supervised_frame(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    exogenous_cols: Sequence[str] | None = None,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    if horizon < 1:
        raise ValueError("Horizon must be >= 1")

    work = df.copy()
    feature_cols: List[str] = []

    for lag in lags:
        col = f"lag_{lag}"
        work[col] = work[target_col].shift(lag)
        feature_cols.append(col)

    for window in rolling_windows:
        mean_col = f"roll_mean_{window}"
        std_col = f"roll_std_{window}"
        shifted = work[target_col].shift(1)
        work[mean_col] = shifted.rolling(window).mean()
        work[std_col] = shifted.rolling(window).std()
        feature_cols.extend([mean_col, std_col])

    for col in exogenous_cols or []:
        if col == date_col or col == target_col:
            continue
        if col not in work.columns:
            raise ValueError(f"Exogenous column not found in data: {col}")
        feature_cols.append(col)

    target_future_col = "target_future"
    work[target_future_col] = work[target_col].shift(-horizon)

    selected_cols = [date_col] + feature_cols + [target_future_col]
    supervised = work[selected_cols].dropna().reset_index(drop=True)
    return supervised, feature_cols


def build_feature_vector_from_history(
    history_levels: Sequence[float],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    exogenous_values: Dict[str, float] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> Dict[str, float]:
    history = [float(x) for x in history_levels]
    if not history:
        raise ValueError("history_levels cannot be empty")

    required = max(max(lags), max(rolling_windows))
    if len(history) < required:
        raise ValueError(f"Need at least {required} observations in history_levels")

    features: Dict[str, float] = {}
    for lag in lags:
        features[f"lag_{lag}"] = float(history[-lag])

    for window in rolling_windows:
        chunk = np.array(history[-window:], dtype=float)
        features[f"roll_mean_{window}"] = float(chunk.mean())
        features[f"roll_std_{window}"] = float(chunk.std(ddof=1)) if len(chunk) > 1 else 0.0

    for key, value in (exogenous_values or {}).items():
        features[str(key)] = float(value)

    if feature_columns:
        missing = [col for col in feature_columns if col not in features]
        if missing:
            raise ValueError(f"Missing values for feature columns: {missing}")

    return features
