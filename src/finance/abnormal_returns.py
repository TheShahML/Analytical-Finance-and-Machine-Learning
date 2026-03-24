"""Abnormal-return helpers for event-study scaffolding."""

from __future__ import annotations

import pandas as pd


def compute_abnormal_return(
    realized_return: pd.Series, benchmark_return: pd.Series
) -> pd.Series:
    """Compute simple abnormal returns as realized minus benchmark return."""

    return realized_return - benchmark_return


def compute_cumulative_abnormal_return(abnormal_returns: pd.Series) -> float:
    """Sum abnormal returns over an event window.

    TODO:
    - confirm whether arithmetic or compounding conventions should be used
    - align this helper with the final event-study specification
    """

    return float(abnormal_returns.sum())
