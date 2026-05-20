from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order


@dataclass
class VARFit:
    model: object
    lag_order: int
    ic: str


@dataclass
class VECMFit:
    model: object
    k_ar_diff: int
    coint_rank: int
    deterministic: str


def adf_pvalues(df: pd.DataFrame) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for col in df.columns:
        series = df[col].dropna().astype(float)
        if series.empty:
            results[col] = float("nan")
            continue
        try:
            results[col] = float(adfuller(series, autolag="AIC")[1])
        except Exception:
            results[col] = float("nan")
    return results


def select_var_fit(endog: pd.DataFrame, maxlags: int = 12, ic: str = "aic") -> VARFit:
    model = VAR(endog)
    try:
        order = model.select_order(maxlags=maxlags)
        lag_order = int(getattr(order, ic))
    except Exception:
        lag_order = 1

    if lag_order < 1:
        lag_order = 1

    result = None
    for lag in range(lag_order, 0, -1):
        try:
            result = model.fit(lag)
            lag_order = lag
            break
        except Exception:
            continue
    if result is None:
        result = model.fit(1)
        lag_order = 1
    return VARFit(model=result, lag_order=lag_order, ic=ic)


def select_vecm_fit(
    endog: pd.DataFrame,
    maxlags: int = 12,
    deterministic: str = "co",
    rank_alpha: float = 0.05,
) -> VECMFit:
    if endog.shape[1] < 2:
        raise ValueError("VECM requires at least 2 endogenous variables")
    order = select_order(endog, maxlags=maxlags, deterministic=deterministic)
    k_ar_diff = int(order.aic) if order.aic is not None else max(1, maxlags // 2)
    k_ar_diff = max(1, k_ar_diff)
    johansen = coint_johansen(endog, det_order=0, k_ar_diff=k_ar_diff)
    trace_stat = johansen.lr1
    crit = johansen.cvt[:, 1]  # 5% column
    coint_rank = int(np.sum(trace_stat > crit))
    coint_rank = max(1, min(coint_rank, endog.shape[1] - 1))
    model = VECM(endog, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
    result = model.fit()
    return VECMFit(model=result, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)


def var_forecast(var_fit: VARFit, history: np.ndarray, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    lag = var_fit.lag_order
    if history.shape[0] < lag:
        raise ValueError(f"Need at least {lag} rows of history for VAR forecast")
    return var_fit.model.forecast(history[-lag:], steps=steps)


def vecm_forecast(vecm_fit: VECMFit, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return vecm_fit.model.predict(steps=steps)
