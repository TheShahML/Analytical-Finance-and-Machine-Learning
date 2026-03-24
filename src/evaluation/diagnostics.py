"""Diagnostic helpers for feature tables and analysis samples."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def summarize_feature_missingness(
    df: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return missingness rates for selected columns."""

    target_columns = list(columns or df.columns)
    summary = pd.DataFrame(
        {
            "column": target_columns,
            "missing_count": [int(df[column].isna().sum()) for column in target_columns],
            "missing_rate": [float(df[column].isna().mean()) for column in target_columns],
        }
    )
    return summary.sort_values("missing_rate", ascending=False).reset_index(drop=True)
