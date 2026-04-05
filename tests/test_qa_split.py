"""Tests for transcript Q&A splitting helpers."""

from __future__ import annotations

import pandas as pd

from src.data.load_transcript_components import component_data_supports_qa_split
from src.data.qa_split import (
    flag_suspicious_qa_pairs,
    pair_questions_responses,
    split_analyst_executive,
    split_prepared_qa,
    summarize_qa_pair_quality,
    validate_qa_split,
)


def _sample_component_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transcriptid": [1, 1, 1, 1],
            "transcriptcomponenttypeid": [2, 3, 3, 3],
            "speakertypeid": [1, 3, 1, 1],
            "componenttext": [
                "Welcome to the call.",
                "How large is the repurchase authorization?",
                "The board approved a $2 billion program.",
                "We expect to finish it over the next year.",
            ],
        }
    )


def test_split_prepared_qa_filters_component_types() -> None:
    prepared, qa = split_prepared_qa(_sample_component_df())

    assert len(prepared) == 1
    assert len(qa) == 3


def test_split_analyst_executive_filters_speaker_types() -> None:
    _, qa = split_prepared_qa(_sample_component_df())
    analyst, executive = split_analyst_executive(qa)

    assert len(analyst) == 1
    assert len(executive) == 2


def test_pair_questions_responses_matches_question_to_next_response() -> None:
    _, qa = split_prepared_qa(_sample_component_df())

    pairs = pair_questions_responses(qa)

    assert len(pairs) == 1
    assert pairs.loc[0, "question_text"] == "How large is the repurchase authorization?"
    assert pairs.loc[0, "response_text"] == "The board approved a $2 billion program."


def test_flag_suspicious_qa_pairs_marks_operator_and_overlong_questions() -> None:
    pairs = pd.DataFrame(
        {
            "question_text": [
                "<strong>Operator</strong> Good morning",
                "word " * 260,
                "How large is the repurchase authorization?",
            ],
            "response_text": [
                "Please go ahead.",
                "Thanks for the question.",
                "The board approved a $2 billion program.",
            ],
        }
    )

    flagged = flag_suspicious_qa_pairs(pairs)

    assert flagged["is_suspicious"].tolist() == [True, True, False]
    assert flagged["question_contains_operator_tag"].tolist() == [True, False, False]
    assert flagged["question_too_long"].tolist() == [False, True, False]


def test_summarize_qa_pair_quality_reports_share() -> None:
    pairs = pd.DataFrame(
        {
            "question_text": ["<strong>Operator</strong> Good morning", "What is the buyback amount?"],
            "response_text": ["Please go ahead.", "It is $1 billion."],
        }
    )

    summary = summarize_qa_pair_quality(pairs)

    assert summary["pair_count"] == 2
    assert summary["suspicious_pairs"] == 1
    assert summary["suspicious_share"] == 0.5


def test_validate_qa_split_reports_core_counts() -> None:
    result = validate_qa_split(_sample_component_df(), sample_n=2)

    assert result["prepared_rows"] == 1
    assert result["qa_rows"] == 3
    assert result["analyst_question_rows"] == 1
    assert result["executive_response_rows"] == 2


def test_component_data_supports_qa_split_requires_speaker_and_component_ids() -> None:
    assert component_data_supports_qa_split(_sample_component_df()) is True
    assert (
        component_data_supports_qa_split(pd.DataFrame({"transcriptid": [1], "componenttext": ["text"]}))
        is False
    )
