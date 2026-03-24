"""Loaders and schema checks for returns data used in finance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_RETURNS_COLUMNS = ["permno", "date", "ret"]


def load_returns_data(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load returns data from CSV or Parquet."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Returns file not found: {dataset_path}")

    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path, columns=columns)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path, usecols=columns)

    raise ValueError("Unsupported returns file type. Use CSV or Parquet.")


def validate_returns_columns(
    columns: Sequence[str], required_columns: Sequence[str] | None = None
) -> list[str]:
    """Return required returns columns that are missing."""

    required = list(required_columns or DEFAULT_RETURNS_COLUMNS)
    observed = set(columns)
    return [column for column in required if column not in observed]
