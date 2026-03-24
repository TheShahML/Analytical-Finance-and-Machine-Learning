"""Conservative text preprocessing helpers for exploratory transcript analysis."""

from __future__ import annotations

import re

import pandas as pd


def basic_clean_text(text: str) -> str:
    """Apply light normalization without aggressively stripping finance terms."""

    cleaned = str(text).lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def prepare_corpus(
    df: pd.DataFrame,
    *,
    text_col: str = "full_transcript_text",
    output_col: str = "clean_text",
    min_characters: int = 100,
) -> pd.DataFrame:
    """Return a copy with lightly cleaned text and empty rows removed."""

    if text_col not in df.columns:
        raise KeyError(f"Expected text column `{text_col}`.")

    result = df.copy()
    result[output_col] = result[text_col].fillna("").astype(str).map(basic_clean_text)
    result = result.loc[result[output_col].str.len() >= min_characters].copy()
    return result.reset_index(drop=True)
