"""Interfaces for embedding-based transcript representations."""

from __future__ import annotations

import pandas as pd


def build_transcript_embeddings(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Placeholder for embedding generation.

    Intended future uses:
    - semantic clustering
    - nearest-neighbor inspection
    - topic-conditioned style exploration
    """

    raise NotImplementedError("Embedding generation is reserved for a later implementation pass.")
