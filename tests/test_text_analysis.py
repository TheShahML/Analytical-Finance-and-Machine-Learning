"""Tests for Task 2 keyword and topic-exploration helpers."""

from __future__ import annotations

import pandas as pd

from src.features.keyword_counts import (
    build_keyword_feature_table,
    build_term_frequency_table,
    keyword_frequency_by_year,
    keyword_frequency_summary,
)
from src.features.text_preprocessing import basic_clean_text, prepare_corpus
from src.features.topic_modeling import (
    assign_dominant_topic,
    extract_top_words_per_topic,
    fit_lda_topic_model,
    prepare_documents_for_topic_modeling,
    select_topic_examples,
    sample_topic_documents,
    summarize_topic_prevalence,
)


def test_basic_clean_text_normalizes_case_and_spacing() -> None:
    assert basic_clean_text("R&D   Growth!!") == "r and d growth"


def test_prepare_corpus_filters_short_rows() -> None:
    df = pd.DataFrame({"full_transcript_text": ["short", "This transcript has enough text to survive. " * 10]})
    result = prepare_corpus(df, min_characters=20)
    assert len(result) == 1


def test_keyword_summary_and_year_rollup_work() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1, 2],
            "call_date": ["2020-01-01", "2021-01-01"],
            "full_transcript_text": [
                "We updated guidance and dividend plans.",
                "Research and development spend increased.",
            ],
        }
    )
    features = build_keyword_feature_table(
        df,
        {
            "guidance": ["guidance"],
            "dividends": ["dividend"],
            "r_and_d": ["research and development"],
        },
    )
    summary = keyword_frequency_summary(features)
    by_year = keyword_frequency_by_year(df, features)

    assert set(summary["keyword_theme"]) == {"guidance", "dividends", "r_and_d"}
    assert set(by_year["year"]) == {2020, 2021}


def test_term_frequency_table_returns_expected_columns() -> None:
    df = pd.DataFrame(
        {"full_transcript_text": ["guidance guidance outlook", "dividend outlook"]}
    )
    result = build_term_frequency_table(df, max_features=10, min_df=1, max_df=1.0, stop_words=None)
    assert {"term", "total_count", "document_frequency", "share_of_documents"} <= set(result.columns)


def test_topic_modeling_pipeline_runs_on_small_documents() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1, 2, 3, 4],
            "call_date": ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"],
            "full_transcript_text": [
                "guidance outlook revenue margin demand " * 30,
                "dividend capital return repurchase cash flow " * 30,
                "cloud software platform customer product growth " * 30,
                "credit loan deposit margin bank reserve " * 30,
            ],
        }
    )
    documents = prepare_documents_for_topic_modeling(
        df,
        extra_columns=["call_date"],
        min_characters=50,
    )
    sampled = sample_topic_documents(documents, max_documents=3, date_column="call_date", random_state=1)
    assert len(sampled) == 3

    model_result = fit_lda_topic_model(
        documents,
        n_topics=2,
        max_features=50,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        stop_words=None,
        max_iter=5,
    )
    top_words = extract_top_words_per_topic(
        model_result["model"],
        model_result["vectorizer"],
        n_top_words=5,
    )
    assignments = assign_dominant_topic(
        model_result["documents_used"],
        model_result["document_topic_matrix"],
    )
    prevalence = summarize_topic_prevalence(assignments, date_column="call_date")
    examples = select_topic_examples(
        assignments.assign(full_transcript_text=df["full_transcript_text"].iloc[: len(assignments)].values),
        n_examples_per_topic=1,
    )

    assert len(top_words) == 2
    assert "dominant_topic" in assignments.columns
    assert not prevalence["overall"].empty
    assert not examples.empty
