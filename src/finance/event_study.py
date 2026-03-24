"""Event-study configuration helpers and future estimation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class EventWindow:
    """Configuration for a relative-day event window."""

    start: int
    end: int


def build_relative_day_index(window: EventWindow) -> list[int]:
    """Return the ordered relative trading-day index for an event window."""

    if window.start > window.end:
        raise ValueError("Event window start must be less than or equal to end.")
    return list(range(window.start, window.end + 1))


def validate_event_study_inputs(
    df: pd.DataFrame,
    required_columns: Sequence[str] = ("permno", "date", "event_date", "ret"),
) -> list[str]:
    """Return missing columns for an event-study input table."""

    observed = set(df.columns)
    return [column for column in required_columns if column not in observed]


def run_event_study(
    event_data: pd.DataFrame,
    estimation_window: EventWindow = EventWindow(start=-120, end=-20),
    event_window: EventWindow = EventWindow(start=-1, end=1),
) -> pd.DataFrame:
    """Placeholder for a future event-study implementation."""

    raise NotImplementedError("Event-study estimation is reserved for a later phase.")
