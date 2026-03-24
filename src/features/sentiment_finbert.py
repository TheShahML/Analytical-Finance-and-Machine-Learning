"""Interfaces for transcript sentiment scoring with finance-oriented transformers."""

from __future__ import annotations

import pandas as pd


def load_finbert_pipeline(model_name: str = "ProsusAI/finbert"):
    """Load the sentiment model pipeline.

    TODO:
    - confirm package versions and runtime requirements
    - decide whether sentiment should run at sentence level or transcript level
    - cache model downloads for reproducibility
    """

    raise NotImplementedError("FinBERT loading is intentionally deferred in this scaffold.")


def score_finbert_sentiment(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Return transcript-level sentiment outputs.

    Expected output columns will likely include:
    - transcript identifier
    - positive score
    - negative score
    - neutral score
    - optional dominant-label field
    """

    raise NotImplementedError("Sentiment scoring is reserved for the next implementation pass.")
