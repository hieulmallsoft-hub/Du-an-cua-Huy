from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def load_adjacent_metadata(data_path: str | Path) -> Dict:
    meta_path = Path(data_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def parse_optional_columns(value: str | None) -> List[str]:
    if value is None:
        return []
    cols = [x.strip() for x in value.split(",") if x.strip()]
    return cols


def load_series(
    data_path: str | Path,
    date_col: str,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    extras = list(feature_cols or [])
    required = {date_col, target_col, *extras}
    missing = required - set(df.columns)
    diff_col = f"{target_col}_diff"
    if diff_col in missing:
        df[diff_col] = pd.to_numeric(df[target_col], errors="coerce").diff().fillna(0.0)
        missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[date_col, target_col, *extras]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    for col in extras:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
    if extras:
        df = df.dropna(subset=extras).reset_index(drop=True)
    duplicate_dates = int(df[date_col].duplicated().sum())
    if duplicate_dates:
        raise ValueError(
            f"Data contains {duplicate_dates} duplicate dates in '{date_col}'. "
            "Use one consistent daily statistic before training."
        )
    return df
