"""Tests for seed-level style scoring helpers."""

from __future__ import annotations

import pandas as pd

from src.features.style_scoring import build_style_score_column_names, score_style_dictionary_features


def test_build_style_score_column_names_uses_standard_suffix() -> None:
    columns = build_style_score_column_names(["hedging", "directness"])
    assert columns == ["hedging_per_1k_words", "directness_per_1k_words"]


def test_score_style_dictionary_features_returns_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "transcriptid": [1],
            "full_transcript_text": ["We believe we may clarify our assumptions clearly."],
        }
    )
    scored = score_style_dictionary_features(df)
    assert "hedging_per_1k_words" in scored.columns
    assert "transparency_per_1k_words" in scored.columns
    assert scored.loc[0, "hedging_per_1k_words"] > 0
