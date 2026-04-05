"""Tests for buyback-event detection helpers."""

from __future__ import annotations

import pandas as pd

from src.data.buyback_events import (
    extract_buyback_sentences,
    identify_buyback_transcripts,
    match_buyback_events_to_transcripts,
)


def test_identify_buyback_transcripts_filters_rows_with_keyword_mentions() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1, 2],
            "full_transcript_text": [
                "We expanded the share repurchase program this quarter.",
                "We discussed cloud demand and margin expansion.",
            ],
        }
    )

    result = identify_buyback_transcripts(df)

    assert result["transcriptid"].tolist() == [1]


def test_extract_buyback_sentences_returns_matching_sentences_only() -> None:
    text = (
        "Revenue grew strongly. "
        "We authorized a $2 billion buyback program through 2027. "
        "The dividend remains unchanged."
    )

    sentences = extract_buyback_sentences(text)

    assert sentences == ["We authorized a $2 billion buyback program through 2027."]


def test_match_buyback_events_to_transcripts_links_nearest_call() -> None:
    key_dev_df = pd.DataFrame(
        {
            "companyid": [10],
            "event_date": ["2024-02-02"],
            "event_type": ["Share Repurchase"],
        }
    )
    transcripts_df = pd.DataFrame(
        {
            "companyid": [10, 10],
            "transcriptid": [100, 101],
            "call_date": ["2024-02-01", "2024-02-05"],
        }
    )

    matched = match_buyback_events_to_transcripts(key_dev_df, transcripts_df, window_days=3)

    assert matched.loc[0, "transcriptid"] == 100
    assert matched.loc[0, "date_distance_days"] == 1
