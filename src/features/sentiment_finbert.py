"""Backward-compatible exports for FinBERT sentiment helpers."""

from __future__ import annotations

from src.features.finbert_sentiment import (
    DEFAULT_FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    aggregate_sentiment,
    bucket_sentiment,
    load_finbert_pipeline,
    score_finbert_sentiment,
    score_sentences,
    score_transcript_sections,
    split_text_into_sentences,
)

__all__ = [
    "DEFAULT_FINBERT_BATCH_SIZE",
    "FINBERT_MODEL_NAME",
    "aggregate_sentiment",
    "bucket_sentiment",
    "load_finbert_pipeline",
    "score_finbert_sentiment",
    "score_sentences",
    "score_transcript_sections",
    "split_text_into_sentences",
]
