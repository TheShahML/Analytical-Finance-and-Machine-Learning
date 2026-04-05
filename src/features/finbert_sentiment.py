"""Sentence-level FinBERT sentiment scoring utilities."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

import numpy as np
import pandas as pd


FINBERT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_FINBERT_BATCH_SIZE = 64

_LABEL_TO_SIGNED_SCORE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}


def split_text_into_sentences(text: object) -> list[str]:
    """Split transcript text into sentence-like chunks for model scoring."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def resolve_torch_device(device: str | None = None) -> str:
    """Return a supported torch device name."""

    requested = device or "cuda"
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only in runtime environments without torch
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@lru_cache(maxsize=2)
def load_finbert_pipeline(
    model_name: str = FINBERT_MODEL_NAME,
    device: str = "cuda",
    batch_size: int = DEFAULT_FINBERT_BATCH_SIZE,
):
    """Load and cache the FinBERT sentiment pipeline."""

    from transformers import pipeline

    resolved_device = resolve_torch_device(device)
    pipeline_device: int | str = resolved_device
    if resolved_device == "cpu":
        pipeline_device = -1

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        device=pipeline_device,
        batch_size=batch_size,
    )


def _score_to_signed_value(label: str, confidence: float) -> float:
    direction = _LABEL_TO_SIGNED_SCORE.get(str(label).lower(), 0.0)
    return float(direction * confidence)


def score_sentences(
    text: object,
    device: str = "cuda",
    batch_size: int = DEFAULT_FINBERT_BATCH_SIZE,
    model_name: str = FINBERT_MODEL_NAME,
    pipeline_obj: Any | None = None,
) -> list[dict[str, Any]]:
    """Run sentence-level FinBERT scoring on a transcript or passage."""

    sentences = split_text_into_sentences(text)
    if not sentences:
        return []

    sentiment_pipeline = pipeline_obj or load_finbert_pipeline(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    predictions = sentiment_pipeline(sentences, batch_size=batch_size, truncation=True, max_length=512)

    results: list[dict[str, Any]] = []
    for sentence, prediction in zip(sentences, predictions):
        label = str(prediction["label"]).lower()
        score = float(prediction["score"])
        results.append(
            {
                "text": sentence,
                "label": label,
                "score": score,
                "signed_score": _score_to_signed_value(label, score),
            }
        )
    return results


def aggregate_sentiment(
    sentence_scores: list[dict[str, Any]],
    method: str = "mean",
) -> float:
    """Aggregate sentence-level signed sentiment scores into one statistic."""

    if not sentence_scores:
        return float("nan")

    values = np.array([float(item.get("signed_score", 0.0)) for item in sentence_scores], dtype=float)
    if method == "mean":
        return float(values.mean())
    if method == "p10":
        return float(np.nanpercentile(values, 10))
    if method == "min":
        return float(values.min())
    raise ValueError(f"Unsupported aggregation method: {method}")


def score_transcript_sections(
    transcript: object,
    prep_text: object,
    qa_text: object,
    buyback_sentences: list[str] | tuple[str, ...],
    *,
    device: str = "cuda",
    batch_size: int = DEFAULT_FINBERT_BATCH_SIZE,
    model_name: str = FINBERT_MODEL_NAME,
    pipeline_obj: Any | None = None,
) -> dict[str, float]:
    """Score multiple transcript scopes and return summary statistics."""

    scopes = {
        "full_transcript": transcript,
        "prepared_remarks": prep_text,
        "qa": qa_text,
        "buyback": " ".join(buyback_sentences),
    }
    output: dict[str, float] = {}

    for scope_name, scope_text in scopes.items():
        sentence_scores = score_sentences(
            scope_text,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            pipeline_obj=pipeline_obj,
        )
        output[f"{scope_name}_sentence_count"] = float(len(sentence_scores))
        output[f"{scope_name}_sentiment_mean"] = aggregate_sentiment(sentence_scores, method="mean")
        output[f"{scope_name}_sentiment_p10"] = aggregate_sentiment(sentence_scores, method="p10")
        output[f"{scope_name}_sentiment_min"] = aggregate_sentiment(sentence_scores, method="min")

    output["buyback_sentiment_gap"] = (
        output["buyback_sentiment_mean"] - output["full_transcript_sentiment_mean"]
    )
    return output


def _safe_group_bucket(
    values: pd.Series,
    labels: list[str],
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = numeric.dropna()
    if valid.empty:
        return result

    ranks = valid.rank(method="first")
    unique_count = int(valid.nunique(dropna=True))
    q = min(len(labels), unique_count) if unique_count > 0 else 0
    if q <= 1:
        result.loc[valid.index] = labels[len(labels) // 2]
        return result

    if q == 2:
        bucket_labels = [labels[0], labels[-1]]
    else:
        bucket_labels = labels
    bucketed = pd.qcut(ranks, q=q, labels=bucket_labels, duplicates="drop")
    result.loc[valid.index] = bucketed.astype(str)
    return result


def bucket_sentiment(
    sentiment_series: pd.Series,
    method: str = "tercile",
    groupby: pd.Series | None = None,
) -> pd.Categorical:
    """Bucket sentiment into Negative / Neutral / Positive terciles."""

    if method != "tercile":
        raise ValueError("Only tercile bucketing is currently supported.")

    labels = ["Negative", "Neutral", "Positive"]
    if groupby is None:
        bucketed = _safe_group_bucket(sentiment_series, labels)
    else:
        group_index = pd.Series(groupby, index=sentiment_series.index)
        bucketed = sentiment_series.groupby(group_index).transform(
            lambda group: _safe_group_bucket(group, labels)
        )
    return pd.Categorical(bucketed, categories=labels, ordered=True)


def score_finbert_sentiment(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
    *,
    device: str = "cuda",
    batch_size: int = DEFAULT_FINBERT_BATCH_SIZE,
    model_name: str = FINBERT_MODEL_NAME,
) -> pd.DataFrame:
    """Backward-compatible transcript-level FinBERT scoring wrapper."""

    if text_column not in df.columns or id_column not in df.columns:
        raise KeyError(f"Expected columns `{id_column}` and `{text_column}`.")

    sentiment_pipeline = load_finbert_pipeline(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    rows: list[dict[str, Any]] = []
    for row in df[[id_column, text_column]].itertuples(index=False):
        record_id = getattr(row, id_column)
        text = getattr(row, text_column)
        sentence_scores = score_sentences(
            text,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            pipeline_obj=sentiment_pipeline,
        )
        rows.append(
            {
                id_column: record_id,
                "finbert_sentiment_mean": aggregate_sentiment(sentence_scores, method="mean"),
                "finbert_sentiment_p10": aggregate_sentiment(sentence_scores, method="p10"),
                "finbert_sentiment_min": aggregate_sentiment(sentence_scores, method="min"),
                "finbert_sentence_count": len(sentence_scores),
            }
        )
    return pd.DataFrame(rows)
