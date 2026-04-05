"""Helpers for event-window calculations from wide `ret_t...` columns."""

from __future__ import annotations

import pandas as pd


def _return_column(relative_day: int) -> str:
    return f"ret_t{relative_day}"


def available_relative_days(df: pd.DataFrame) -> list[int]:
    """Return all relative days present in wide return columns."""

    days: list[int] = []
    for column in df.columns:
        if column.startswith("ret_t"):
            suffix = column.replace("ret_t", "", 1)
            try:
                days.append(int(suffix))
            except ValueError:
                continue
    return sorted(days)


def compute_car_from_wide_returns(
    df: pd.DataFrame,
    start_day: int = 1,
    end_day: int = 3,
) -> pd.Series:
    """Compute a simple cumulative return over a post-event window."""

    if start_day > end_day:
        raise ValueError("start_day must be less than or equal to end_day.")

    columns = [_return_column(day) for day in range(start_day, end_day + 1)]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing wide return columns required for CAR computation: {missing}")

    return df.loc[:, columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
