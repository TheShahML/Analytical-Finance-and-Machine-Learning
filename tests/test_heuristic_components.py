"""Tests for heuristic transcript component extraction."""

from __future__ import annotations

import pandas as pd

from src.data.heuristic_components import (
    ANALYST_SPEAKER_TYPE,
    EXECUTIVE_SPEAKER_TYPE,
    OPERATOR_SPEAKER_TYPE,
    PREPARED_REMARKS_TYPE,
    QA_TYPE,
    build_component_dataset_from_transcripts,
    extract_transcript_components_from_text,
    split_transcript_paragraphs,
)


def test_split_transcript_paragraphs_collapses_whitespace() -> None:
    text = "Hello world.\n\nThis is   a test.\n\n\nFinal block."
    parts = split_transcript_paragraphs(text)
    assert parts == ["Hello world.", "This is a test.", "Final block."]


def test_split_transcript_paragraphs_breaks_inline_speaker_tags_and_prompts() -> None:
    text = (
        "<strong>Operator</strong> Good morning everyone. "
        "<strong>CEO</strong> We had a strong quarter. "
        "Our next question comes from the line of Jane Doe with Example Bank. "
        "Can you discuss the repurchase plan? "
        "<strong>CEO</strong> Yes, it is funded by free cash flow."
    )

    parts = split_transcript_paragraphs(text)

    assert len(parts) >= 4
    assert parts[0].startswith("<strong>Operator</strong>")
    assert any(part.startswith("Our next question comes from") for part in parts)
    assert parts[-1].startswith("<strong>CEO</strong>")


def test_extract_transcript_components_identifies_prepared_and_qa_blocks() -> None:
    text = (
        "Good morning and welcome.\n\n"
        "We delivered strong revenue growth this quarter.\n\n"
        "Thanks, Rich. We'd like to open it up for questions.\n\n"
        "[Operator Instructions] Your first question comes from the line of Jane Doe with Example Bank.\n\n"
        "Can you talk about the repurchase authorization?\n\n"
        "Yes, we approved a $2 billion program over the next year funded by free cash flow.\n"
    )

    result = extract_transcript_components_from_text(text, transcript_id=1)

    assert result["transcriptcomponenttypeid"].tolist() == [2, 2, 2, 3, 3, 3, 3, 3]
    assert result["speakertypeid"].tolist() == [1, 1, 1, 1, 0, 0, 3, 1]
    assert result.loc[6, "speakername"] == "Jane Doe"


def test_build_component_dataset_from_transcripts_preserves_metadata() -> None:
    transcripts = pd.DataFrame(
        {
            "transcriptid": [1],
            "companyid": [10],
            "companyname": ["Example Co"],
            "ticker": ["EXM"],
            "call_date": ["2024-01-31"],
            "full_transcript_text": [
                "Prepared remarks.\n\n"
                "Your next question comes from the line of Jane Doe with Example Bank.\n\n"
                "What is the buyback amount?\n\n"
                "Yes, it is $1 billion."
            ],
        }
    )

    result = build_component_dataset_from_transcripts(transcripts)

    assert set(["companyid", "companyname", "ticker", "call_date"]) <= set(result.columns)
    assert PREPARED_REMARKS_TYPE in set(result["transcriptcomponenttypeid"])
    assert QA_TYPE in set(result["transcriptcomponenttypeid"])
    assert ANALYST_SPEAKER_TYPE in set(result["speakertypeid"])
    assert EXECUTIVE_SPEAKER_TYPE in set(result["speakertypeid"])
    assert OPERATOR_SPEAKER_TYPE in set(result["speakertypeid"])
