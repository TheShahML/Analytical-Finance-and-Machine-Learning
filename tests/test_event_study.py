"""Tests for event-study helpers."""

from __future__ import annotations

import pandas as pd

from src.finance.abnormal_returns import compute_abnormal_return
from src.finance.event_study import (
    EventWindow,
    IMAGE_SPEC_ESTIMATION_WINDOW,
    IMAGE_SPEC_ROBUSTNESS_WINDOWS,
    build_relative_day_index,
    compute_abnormal_returns_from_wide_returns,
    compute_caar_by_bins,
    compute_expected_return_from_wide_returns,
    run_event_study_from_wide_returns,
    run_event_study,
)


def test_build_relative_day_index_includes_both_endpoints() -> None:
    index = build_relative_day_index(EventWindow(start=-1, end=1))
    assert index == [-1, 0, 1]


def test_event_study_default_estimation_window_matches_buyback_spec() -> None:
    assert run_event_study.__defaults__[0] == EventWindow(start=-15, end=-3)


def test_image_spec_constants_match_slides() -> None:
    assert IMAGE_SPEC_ESTIMATION_WINDOW == EventWindow(start=-15, end=-3)
    assert IMAGE_SPEC_ROBUSTNESS_WINDOWS == (
        EventWindow(start=1, end=3),
        EventWindow(start=1, end=5),
    )


def test_compute_abnormal_return_is_difference_between_series() -> None:
    realized = pd.Series([0.02, -0.01])
    benchmark = pd.Series([0.01, -0.02])
    abnormal = compute_abnormal_return(realized, benchmark)
    assert list(abnormal.round(4)) == [0.01, 0.01]


def test_run_event_study_mean_model_computes_event_level_car() -> None:
    rows = []
    for relative_day in range(-5, 4):
        rows.append(
            {
                "permno": 10001,
                "date": pd.Timestamp("2024-01-10") + pd.Timedelta(days=relative_day),
                "event_date": pd.Timestamp("2024-01-10"),
                "ret": 0.01 if relative_day < 0 else 0.03,
            }
        )
    event_df = pd.DataFrame(rows)

    result = run_event_study(
        event_df,
        estimation_window=EventWindow(start=-5, end=-1),
        event_window=EventWindow(start=0, end=1),
    )

    assert result["abnormal_return"].round(4).tolist() == [0.02, 0.02]
    assert result["car"].round(4).tolist() == [0.04, 0.04]


def test_compute_caar_by_bins_summarizes_group_means() -> None:
    df = pd.DataFrame(
        {
            "sentiment_bucket": ["Positive", "Positive", "Negative"],
            "clarity_bucket": ["High", "High", "Low"],
            "car": [0.03, 0.01, -0.02],
        }
    )

    result = compute_caar_by_bins(df, ["sentiment_bucket", "clarity_bucket"])

    positive_high = result.loc[
        (result["sentiment_bucket"] == "Positive") & (result["clarity_bucket"] == "High")
    ].iloc[0]
    assert positive_high["n"] == 2
    assert round(positive_high["caar"], 4) == 0.02


def test_compute_expected_return_from_wide_returns_uses_mean_model() -> None:
    df = pd.DataFrame(
        {
            "ret_t-15": [0.01],
            "ret_t-14": [0.02],
            "ret_t-13": [0.03],
            "ret_t-12": [0.04],
            "ret_t-11": [0.05],
            "ret_t-10": [0.06],
            "ret_t-9": [0.07],
            "ret_t-8": [0.08],
            "ret_t-7": [0.09],
            "ret_t-6": [0.10],
            "ret_t-5": [0.11],
            "ret_t-4": [0.12],
            "ret_t-3": [0.13],
        }
    )

    expected = compute_expected_return_from_wide_returns(df)

    assert round(expected.iloc[0], 4) == 0.07


def test_run_event_study_from_wide_returns_computes_image_spec_windows() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1],
            "permno": [10001],
            "call_date": ["2024-01-10"],
            **{f"ret_t{day}": [0.01] for day in range(-15, -2)},
            "ret_t1": [0.04],
            "ret_t2": [0.03],
            "ret_t3": [0.02],
            "ret_t4": [0.01],
            "ret_t5": [0.00],
        }
    )

    result = run_event_study_from_wide_returns(df)

    assert round(result.loc[0, "expected_return"], 4) == 0.01
    assert round(result.loc[0, "abnormal_ret_t1"], 4) == 0.03
    assert round(result.loc[0, "car_1_3"], 4) == 0.06
    assert round(result.loc[0, "car_1_5"], 4) == 0.05


def test_compute_abnormal_returns_from_wide_returns_adds_columns() -> None:
    df = pd.DataFrame(
        {
            "ret_t-15": [0.01],
            "ret_t-14": [0.01],
            "ret_t-13": [0.01],
            "ret_t-12": [0.01],
            "ret_t-11": [0.01],
            "ret_t-10": [0.01],
            "ret_t-9": [0.01],
            "ret_t-8": [0.01],
            "ret_t-7": [0.01],
            "ret_t-6": [0.01],
            "ret_t-5": [0.01],
            "ret_t-4": [0.01],
            "ret_t-3": [0.01],
            "ret_t1": [0.03],
        }
    )

    result = compute_abnormal_returns_from_wide_returns(df)

    assert "expected_return" in result.columns
    assert round(result.loc[0, "abnormal_ret_t1"], 4) == 0.02


def test_run_event_study_from_wide_returns_accepts_final_csv_negative_day_columns() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1],
            "permno": [10001],
            "call_date": ["2024-01-10"],
            **{f"ret_t_{abs(day)}": [0.01] for day in range(-15, -2)},
            "ret_t1": [0.04],
            "ret_t2": [0.03],
            "ret_t3": [0.02],
            "ret_t4": [0.01],
            "ret_t5": [0.00],
        }
    )

    result = run_event_study_from_wide_returns(df)

    assert round(result.loc[0, "expected_return"], 4) == 0.01
    assert round(result.loc[0, "abnormal_ret_t1"], 4) == 0.03
    assert round(result.loc[0, "car_1_3"], 4) == 0.06
