"""Binning and portfolio-sort helpers for buyback event analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _group_summary(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    n_obs = int(len(numeric))
    mean_value = float(numeric.mean()) if n_obs else float("nan")
    std_error = float(numeric.std(ddof=1) / np.sqrt(n_obs)) if n_obs > 1 else float("nan")
    has_standard_error = n_obs > 1 and not np.isnan(std_error) and std_error != 0.0
    t_stat = float(mean_value / std_error) if has_standard_error else float("nan")
    p_value = (
        float(stats.ttest_1samp(numeric, popmean=0.0, nan_policy="omit").pvalue)
        if n_obs > 1
        else float("nan")
    )
    return {
        "n": n_obs,
        "mean_car": mean_value,
        "standard_error": std_error,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def create_sentiment_clarity_matrix(
    df: pd.DataFrame,
    sentiment_col: str,
    clarity_col: str,
    car_col: str,
) -> pd.DataFrame:
    """Create a 3x3 sentiment-by-clarity matrix of CAR outcomes."""

    grouped = (
        df.dropna(subset=[sentiment_col, clarity_col, car_col])
        .groupby([sentiment_col, clarity_col], observed=False)[car_col]
        .apply(_group_summary)
    )
    matrix = pd.DataFrame(list(grouped), index=grouped.index).reset_index()
    return matrix


def create_three_way_sort(
    df: pd.DataFrame,
    sentiment_col: str,
    clarity_col: str,
    revenue_col: str,
    car_col: str,
) -> pd.DataFrame:
    """Compute the 3x3x3 sort by sentiment, clarity, and revenue surprise."""

    grouped = (
        df.dropna(subset=[sentiment_col, clarity_col, revenue_col, car_col])
        .groupby([revenue_col, sentiment_col, clarity_col], observed=False)[car_col]
        .apply(_group_summary)
    )
    return pd.DataFrame(list(grouped), index=grouped.index).reset_index()


def compute_spread(
    df: pd.DataFrame,
    high_group: pd.Series | np.ndarray | list[bool],
    low_group: pd.Series | np.ndarray | list[bool],
    car_col: str,
) -> dict[str, float]:
    """Compute the return spread between two selected groups."""

    high = pd.to_numeric(df.loc[pd.Series(high_group, index=df.index), car_col], errors="coerce").dropna()
    low = pd.to_numeric(df.loc[pd.Series(low_group, index=df.index), car_col], errors="coerce").dropna()
    spread = float(high.mean() - low.mean()) if len(high) and len(low) else float("nan")
    test = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit") if len(high) > 1 and len(low) > 1 else None
    return {
        "high_n": int(len(high)),
        "low_n": int(len(low)),
        "spread": spread,
        "t_stat": float(test.statistic) if test is not None else float("nan"),
        "p_value": float(test.pvalue) if test is not None else float("nan"),
    }


def format_results_table(matrix_results: pd.DataFrame) -> pd.DataFrame:
    """Format results for export to the tables directory."""

    formatted = matrix_results.copy()
    for column in ["mean_car", "standard_error", "t_stat", "p_value"]:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(4)
    if "n" in formatted.columns:
        formatted["n"] = pd.to_numeric(formatted["n"], errors="coerce").astype("Int64")
    return formatted
