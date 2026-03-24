"""Tests for event-study scaffolding helpers."""

from __future__ import annotations

import pandas as pd

from src.finance.abnormal_returns import compute_abnormal_return
from src.finance.event_study import EventWindow, build_relative_day_index


def test_build_relative_day_index_includes_both_endpoints() -> None:
    index = build_relative_day_index(EventWindow(start=-1, end=1))
    assert index == [-1, 0, 1]


def test_compute_abnormal_return_is_difference_between_series() -> None:
    realized = pd.Series([0.02, -0.01])
    benchmark = pd.Series([0.01, -0.02])
    abnormal = compute_abnormal_return(realized, benchmark)
    assert list(abnormal.round(4)) == [0.01, 0.01]
