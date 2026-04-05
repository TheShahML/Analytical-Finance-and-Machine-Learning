"""Embedding-based transcript representations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


def _resolve_device(device: str | None = None) -> str:
    requested = device or "cuda"
    try:
        import torch
    except ImportError:  # pragma: no cover - runtime-only fallback
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda",
):
    """Load and cache the default sentence-transformer model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_resolve_device(device))


def build_transcript_embeddings(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    """Encode transcript text into reusable dense embeddings."""

    if text_column not in df.columns or id_column not in df.columns:
        raise KeyError(f"Expected columns `{id_column}` and `{text_column}`.")

    model = load_embedding_model(model_name=model_name, device=device)
    texts = df[text_column].fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return pd.DataFrame(
        {
            id_column: df[id_column].tolist(),
            "embedding": list(np.asarray(embeddings)),
        }
    )
