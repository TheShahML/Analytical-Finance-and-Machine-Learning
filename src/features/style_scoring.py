"""Dictionary-based and model-assisted interfaces for transcript style scoring."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.features.keyword_counts import count_keyword_matches
from src.features.style_dictionary import STYLE_DICTIONARY_SEEDS, get_style_signal_names


def build_style_score_column_names(signal_names: Sequence[str] | None = None) -> list[str]:
    """Return standardized output column names for style scores."""

    names = list(signal_names or get_style_signal_names())
    return [f"{name}_per_1k_words" for name in names]


def score_style_dictionary_features(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Create transcript-level dictionary-based style scores.

    Scores are normalized as occurrences per 1,000 words. These are explicitly
    seed-level prototypes and should be inspected before any empirical use.
    """

    if text_column not in df.columns or id_column not in df.columns:
        raise KeyError(f"Expected columns `{id_column}` and `{text_column}` in transcript data.")

    texts = df[text_column].fillna("").astype(str)
    word_counts = texts.str.split().str.len().clip(lower=1)

    result = df[[id_column]].copy()
    for signal_name, terms in STYLE_DICTIONARY_SEEDS.items():
        raw_counts = texts.apply(lambda text: count_keyword_matches(text, terms))
        result[f"{signal_name}_per_1k_words"] = raw_counts.div(word_counts).mul(1000.0)

    return result


def score_style_model_assisted(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Placeholder for later model-assisted style scoring workflows."""

    raise NotImplementedError("Model-assisted style scoring is reserved for a later phase.")
