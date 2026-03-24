"""Tests for transcript validation and Task 1 cleaned-output helpers."""

from __future__ import annotations

import pandas as pd

from src.data.build_panel import flag_usable_transcripts
from src.data.clean_transcripts import apply_cleaning_pipeline, deduplicate_transcript_events
from src.data.load_transcripts import standardize_transcript_columns
from src.data.validate_transcripts import (
    build_validation_summary,
    check_required_columns,
    summarize_identifier_match_rate,
    summarize_text_length,
    validate_transcript_schema,
)


def test_validate_transcript_schema_detects_missing_required_columns() -> None:
    columns = ["transcriptid", "companyname", "call_date"]
    missing = validate_transcript_schema(columns)
    assert missing == ["ticker", "full_transcript_text"]


def test_build_validation_summary_counts_duplicates_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1, 1, 2],
            "companyname": ["A", "A", "B"],
            "ticker": ["AAA", "AAA", "BBB"],
            "full_transcript_text": ["hello world", None, "text"],
            "call_date": ["2024-01-01", "2024-01-01", None],
        }
    )
    summary = build_validation_summary(df)
    assert summary.row_count == 3
    assert summary.missing_required_columns == []
    assert summary.duplicate_transcript_ids == 1
    assert summary.missing_text_rows == 1
    assert summary.missing_call_date_rows == 1


def test_check_required_columns_marks_presence() -> None:
    df = pd.DataFrame(columns=["transcriptid", "companyname", "ticker", "full_transcript_text"])
    result = check_required_columns(df, required_columns=["transcriptid", "call_date"])
    assert result.to_dict(orient="records") == [
        {"column": "transcriptid", "present": True},
        {"column": "call_date", "present": False},
    ]


def test_standardize_transcript_columns_normalizes_names() -> None:
    df = pd.DataFrame(columns=["Transcript ID", "Call-Date", "Full Transcript Text"])
    standardized = standardize_transcript_columns(df)
    assert list(standardized.columns) == ["transcript_id", "call_date", "full_transcript_text"]


def test_summarize_identifier_match_rate_handles_missing_columns() -> None:
    df = pd.DataFrame({"ticker": ["AAA", None], "permno": [10001, 10002]})
    result = summarize_identifier_match_rate(df, ["ticker", "permno", "gvkey"])
    records = result.to_dict(orient="records")
    assert records[0]["match_rate"] == 0.5
    assert records[1]["match_rate"] == 1.0
    assert records[2]["present"] is False


def test_flag_usable_transcripts_marks_missing_text_or_dates() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1, 2, 3],
            "full_transcript_text": ["text", "", "text"],
            "call_date": ["2024-01-01", "2024-01-02", None],
        }
    )
    flagged = flag_usable_transcripts(df)
    assert flagged["is_usable"].tolist() == [True, False, False]
    assert "missing_or_blank_text" in flagged.loc[1, "exclusion_reason"]
    assert "missing_call_date" in flagged.loc[2, "exclusion_reason"]


def test_summarize_text_length_defaults_to_reported_columns() -> None:
    df = pd.DataFrame(
        {
            "full_transcript_text": ["one two three", "four five"],
            "transcript_length": [13, 9],
            "word_count": [3, 2],
        }
    )
    report = summarize_text_length(df, "full_transcript_text")
    summary_metrics = set(report["summary"]["metric"].tolist())
    assert "reported_character_count" in summary_metrics
    assert "reported_word_count" in summary_metrics
    assert "computed_word_count" not in summary_metrics


def test_deduplicate_transcript_events_keeps_longest_transcript() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "call_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "transcript_length": [100, 250, 90],
            "full_transcript_text": ["short", "longer text", "other"],
            "transcriptid": [1, 2, 3],
        }
    )
    deduped = deduplicate_transcript_events(df)
    assert len(deduped) == 2
    assert set(deduped["transcriptid"]) == {2, 3}


def test_apply_cleaning_pipeline_filters_dates_deduplicates_and_drops_exact_zero_rows() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "call_date": ["2009-12-31", "2010-01-02", "2010-01-02", "2010-01-03"],
            "transcript_length": [50, 200, 100, 100],
            "full_transcript_text": ["old", "keep me", "drop zero return", "keep"],
            "close_price_call_day": [10.0, 10.0, 7.0, 5.0],
            "open_price_next_day": [11.0, 12.0, 7.0, 6.0],
        }
    )
    cleaned, log = apply_cleaning_pipeline(df)
    assert len(cleaned) == 2
    assert cleaned["call_date"].min() >= "2010-01-02"
    assert len(log) == 3
