"""Event-study helpers for earnings-call and buyback announcement analyses."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.finance.abnormal_returns import compute_abnormal_return, compute_cumulative_abnormal_return


@dataclass(frozen=True)
class EventWindow:
    """Configuration for a relative-day event window."""

    start: int
    end: int


IMAGE_SPEC_ESTIMATION_WINDOW = EventWindow(start=-15, end=-3)
IMAGE_SPEC_EVENT_WINDOW = EventWindow(start=1, end=5)
IMAGE_SPEC_ROBUSTNESS_WINDOWS: tuple[EventWindow, ...] = (
    EventWindow(start=1, end=3),
    EventWindow(start=1, end=5),
)


def build_relative_day_index(window: EventWindow) -> list[int]:
    """Return the ordered relative trading-day index for an event window."""

    if window.start > window.end:
        raise ValueError("Event window start must be less than or equal to end.")
    return list(range(window.start, window.end + 1))


def _return_column_name(relative_day: int, prefix: str = "ret_t") -> str:
    return f"{prefix}{relative_day}"


def _return_column_candidates(relative_day: int, prefix: str = "ret_t") -> list[str]:
    if relative_day < 0:
        return [f"{prefix}{relative_day}", f"{prefix}_{abs(relative_day)}"]
    if relative_day == 0:
        return [f"{prefix}0", f"{prefix}_0"]
    return [f"{prefix}{relative_day}", f"{prefix}_{relative_day}"]


def _resolve_return_column(df: pd.DataFrame, relative_day: int, prefix: str = "ret_t") -> str:
    for candidate in _return_column_candidates(relative_day, prefix=prefix):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"Missing return column for relative day {relative_day}; "
        f"tried {_return_column_candidates(relative_day, prefix=prefix)}."
    )


def _parse_relative_day_from_column(column: str, prefix: str = "ret_t") -> int | None:
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_)?(-?\d+)$")
    match = pattern.match(column)
    if not match:
        return None

    day_token = match.group(1)
    if column.startswith(f"{prefix}_") and not day_token.startswith("-"):
        return -int(day_token)
    return int(day_token)


def validate_event_study_inputs(
    df: pd.DataFrame,
    required_columns: Sequence[str] = ("permno", "date", "event_date", "ret"),
) -> list[str]:
    """Return missing columns for a panel-style event-study input table."""

    observed = set(df.columns)
    return [column for column in required_columns if column not in observed]


def _assign_event_id(
    event_data: pd.DataFrame,
    firm_col: str,
    event_date_col: str,
) -> pd.Series:
    event_date = pd.to_datetime(event_data[event_date_col], errors="coerce")
    return event_data[firm_col].astype(str) + "::" + event_date.dt.strftime("%Y-%m-%d")


def _compute_relative_day(group: pd.DataFrame, date_col: str, event_date_col: str) -> pd.Series:
    dates = pd.to_datetime(group[date_col], errors="coerce")
    event_date = pd.to_datetime(group[event_date_col].iloc[0], errors="coerce")
    event_matches = dates.eq(event_date)

    if event_matches.any():
        event_position = int(np.flatnonzero(event_matches.to_numpy())[0])
        return pd.Series(np.arange(len(group)) - event_position, index=group.index)

    return (dates - event_date).dt.days


def _estimate_mean_expected_return(estimation_sample: pd.DataFrame, return_col: str) -> float:
    return float(pd.to_numeric(estimation_sample[return_col], errors="coerce").mean())


def _estimate_market_model_params(
    estimation_sample: pd.DataFrame,
    return_col: str,
    market_return_col: str,
) -> tuple[float, float]:
    sample = (
        estimation_sample[[return_col, market_return_col]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if len(sample) < 3 or sample[market_return_col].nunique() < 2:
        return float(sample[return_col].mean()) if len(sample) else 0.0, 0.0

    beta, alpha = np.polyfit(sample[market_return_col], sample[return_col], deg=1)
    return float(alpha), float(beta)


def compute_expected_return_from_wide_returns(
    df: pd.DataFrame,
    estimation_window: EventWindow = IMAGE_SPEC_ESTIMATION_WINDOW,
    *,
    return_prefix: str = "ret_t",
) -> pd.Series:
    """Compute a mean-model expected return from wide relative-day returns."""

    columns = [
        _resolve_return_column(df, day, prefix=return_prefix)
        for day in build_relative_day_index(estimation_window)
    ]

    return df.loc[:, columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def compute_abnormal_returns_from_wide_returns(
    df: pd.DataFrame,
    estimation_window: EventWindow = IMAGE_SPEC_ESTIMATION_WINDOW,
    *,
    return_prefix: str = "ret_t",
    output_prefix: str = "abnormal_ret_t",
) -> pd.DataFrame:
    """Append expected return plus abnormal-return columns to a wide event table."""

    expected_return = compute_expected_return_from_wide_returns(
        df,
        estimation_window=estimation_window,
        return_prefix=return_prefix,
    )
    result = df.copy()
    result["expected_return"] = expected_return

    for column in [col for col in df.columns if col.startswith(return_prefix)]:
        relative_day = _parse_relative_day_from_column(column, prefix=return_prefix)
        if relative_day is None:
            continue
        realized = pd.to_numeric(df[column], errors="coerce")
        result[_return_column_name(relative_day, prefix=output_prefix)] = compute_abnormal_return(
            realized,
            expected_return,
        )

    return result


def compute_car_from_wide_abnormal_returns(
    df: pd.DataFrame,
    event_window: EventWindow,
    *,
    abnormal_prefix: str = "abnormal_ret_t",
) -> pd.Series:
    """Compute CAR over a post-event window from wide abnormal-return columns."""

    columns = [
        _resolve_return_column(df, day, prefix=abnormal_prefix)
        for day in build_relative_day_index(event_window)
    ]

    abnormal_returns = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return abnormal_returns.sum(axis=1, min_count=1)


def run_event_study_from_wide_returns(
    event_df: pd.DataFrame,
    estimation_window: EventWindow = IMAGE_SPEC_ESTIMATION_WINDOW,
    event_windows: Sequence[EventWindow] = IMAGE_SPEC_ROBUSTNESS_WINDOWS,
    *,
    id_columns: Sequence[str] = ("transcriptid", "permno", "call_date"),
    return_prefix: str = "ret_t",
    abnormal_prefix: str = "abnormal_ret_t",
) -> pd.DataFrame:
    """Run the image-spec mean-model event study on wide return columns."""

    result = compute_abnormal_returns_from_wide_returns(
        event_df,
        estimation_window=estimation_window,
        return_prefix=return_prefix,
        output_prefix=abnormal_prefix,
    )

    for window in event_windows:
        window_label = f"car_{window.start}_{window.end}"
        result[window_label] = compute_car_from_wide_abnormal_returns(
            result,
            event_window=window,
            abnormal_prefix=abnormal_prefix,
        )

    keep_columns = [column for column in id_columns if column in result.columns]
    abnormal_columns = [column for column in result.columns if column.startswith(abnormal_prefix)]
    car_columns = [f"car_{window.start}_{window.end}" for window in event_windows]
    selected_columns = keep_columns + ["expected_return"] + abnormal_columns + car_columns
    return result.loc[:, selected_columns].copy()


def run_event_study(
    event_data: pd.DataFrame,
    estimation_window: EventWindow = IMAGE_SPEC_ESTIMATION_WINDOW,
    event_window: EventWindow = IMAGE_SPEC_EVENT_WINDOW,
    *,
    model: str = "mean",
    firm_col: str = "permno",
    date_col: str = "date",
    event_date_col: str = "event_date",
    return_col: str = "ret",
    market_return_col: str = "market_ret",
    event_id_col: str | None = None,
) -> pd.DataFrame:
    """Estimate abnormal returns and CARs for a panel of daily returns."""

    missing_columns = validate_event_study_inputs(
        event_data,
        required_columns=(firm_col, date_col, event_date_col, return_col),
    )
    if missing_columns:
        raise KeyError(f"Missing required event-study columns: {missing_columns}")

    if model == "market" and market_return_col not in event_data.columns:
        raise KeyError(f"Expected market return column `{market_return_col}` for market-model estimation.")
    if model not in {"mean", "market"}:
        raise ValueError("Supported models are `mean` and `market`.")

    working = event_data.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[event_date_col] = pd.to_datetime(working[event_date_col], errors="coerce")
    if event_id_col is None or event_id_col not in working.columns:
        working["_event_id"] = _assign_event_id(
            working,
            firm_col=firm_col,
            event_date_col=event_date_col,
        )
        event_id_col = "_event_id"

    working = working.sort_values([event_id_col, date_col]).copy()
    relative_day_parts = [
        _compute_relative_day(group, date_col=date_col, event_date_col=event_date_col)
        for _, group in working.groupby(event_id_col, sort=False)
    ]
    working["relative_day"] = pd.concat(relative_day_parts).sort_index()
    working["relative_day"] = pd.to_numeric(working["relative_day"], errors="coerce")

    event_rows: list[pd.DataFrame] = []
    for _, group in working.groupby(event_id_col, sort=False):
        estimation_sample = group.loc[
            group["relative_day"].between(estimation_window.start, estimation_window.end)
        ].copy()
        event_sample = group.loc[
            group["relative_day"].between(event_window.start, event_window.end)
        ].copy()
        if estimation_sample.empty or event_sample.empty:
            continue

        if model == "mean":
            expected = _estimate_mean_expected_return(estimation_sample, return_col=return_col)
            event_sample["expected_return"] = expected
        else:
            alpha, beta = _estimate_market_model_params(
                estimation_sample,
                return_col=return_col,
                market_return_col=market_return_col,
            )
            event_sample["expected_return"] = alpha + beta * pd.to_numeric(
                event_sample[market_return_col], errors="coerce"
            )

        realized = pd.to_numeric(event_sample[return_col], errors="coerce")
        benchmark = pd.to_numeric(event_sample["expected_return"], errors="coerce")
        event_sample["abnormal_return"] = compute_abnormal_return(realized, benchmark)
        car = compute_cumulative_abnormal_return(event_sample["abnormal_return"].dropna())
        event_sample["car"] = car
        event_sample["expected_return_model"] = model
        event_rows.append(event_sample)

    if not event_rows:
        return pd.DataFrame(
            columns=[
                event_id_col,
                firm_col,
                event_date_col,
                "relative_day",
                "expected_return",
                "abnormal_return",
                "car",
                "expected_return_model",
            ]
        )

    return pd.concat(event_rows, ignore_index=True)


def summarize_event_cars(
    event_window_returns: pd.DataFrame,
    *,
    car_col: str = "car",
    event_id_col: str = "_event_id",
    keep_columns: Sequence[str] = ("permno", "event_date", "expected_return_model"),
) -> pd.DataFrame:
    """Collapse panel-style event-window rows to one CAR observation per event."""

    available_keep_columns = [column for column in keep_columns if column in event_window_returns.columns]
    summary_columns = [event_id_col, car_col, *available_keep_columns]
    deduped = event_window_returns.loc[:, summary_columns].drop_duplicates(subset=[event_id_col]).copy()
    return deduped.reset_index(drop=True)


def test_caar_significance(cars: pd.Series | Sequence[float]) -> dict[str, float]:
    """Test whether average CAR differs from zero."""

    numeric = pd.to_numeric(pd.Series(cars), errors="coerce").dropna()
    n_obs = int(len(numeric))
    if n_obs == 0:
        return {
            "n": 0,
            "caar": float("nan"),
            "standard_error": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    caar = float(numeric.mean())
    standard_error = float(numeric.std(ddof=1) / np.sqrt(n_obs)) if n_obs > 1 else float("nan")
    if n_obs > 1:
        test_result = stats.ttest_1samp(numeric, popmean=0.0, nan_policy="omit")
        t_stat = float(test_result.statistic)
        p_value = float(test_result.pvalue)
    else:
        t_stat = float("nan")
        p_value = float("nan")
    return {
        "n": n_obs,
        "caar": caar,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def compute_caar_by_bins(
    car_df: pd.DataFrame,
    bin_columns: Sequence[str],
    *,
    car_col: str = "car",
) -> pd.DataFrame:
    """Compute CAAR and significance metrics by bin combinations."""

    missing = [column for column in [*bin_columns, car_col] if column not in car_df.columns]
    if missing:
        raise KeyError(f"Missing required CAAR columns: {missing}")

    records: list[dict[str, float | str]] = []
    for group_values, group in car_df.dropna(subset=list(bin_columns) + [car_col]).groupby(
        list(bin_columns),
        observed=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        summary = test_caar_significance(group[car_col])
        record = {column: value for column, value in zip(bin_columns, group_values)}
        record.update(summary)
        records.append(record)
    return pd.DataFrame(records)
