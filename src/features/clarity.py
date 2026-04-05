"""Clarity features for buyback-specific executive Q&A responses."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import zscore

from src.config.settings import DATA_DIR


FINANCIAL_VOCAB_EXCLUSIONS: set[str] = {
    "acceleration",
    "acquisition",
    "administration",
    "allocation",
    "approximately",
    "capitalization",
    "collaboration",
    "commercialization",
    "communication",
    "company",
    "competition",
    "compliance",
    "consolidation",
    "consideration",
    "consumer",
    "corporation",
    "customer",
    "depreciation",
    "distribution",
    "diversification",
    "efficiency",
    "environmental",
    "evaluation",
    "execution",
    "financial",
    "foundation",
    "generation",
    "infrastructure",
    "innovation",
    "institutional",
    "integration",
    "international",
    "investment",
    "liquidity",
    "management",
    "manufacturing",
    "opportunity",
    "operating",
    "optimization",
    "organization",
    "performance",
    "portfolio",
    "positioning",
    "productivity",
    "regulatory",
    "relationship",
    "securities",
    "shareholder",
    "strategic",
    "technology",
    "transaction",
    "valuation",
}

DEFAULT_UNCERTAINTY_WORDS: set[str] = {
    "approximately",
    "assume",
    "assumption",
    "believe",
    "could",
    "depends",
    "estimate",
    "maybe",
    "might",
    "possible",
    "potential",
    "uncertain",
}

DEFAULT_WEAK_MODAL_WORDS: set[str] = {
    "can",
    "could",
    "may",
    "might",
    "possibly",
    "should",
}

DEFAULT_LM_DICTIONARY_CANDIDATES = [
    DATA_DIR / "external" / "Loughran-McDonald_MasterDictionary_1993-2024.csv",
    DATA_DIR / "external" / "Loughran-McDonald_MasterDictionary_1993-2024.xlsx",
    DATA_DIR / "external" / "Loughran-McDonald_MasterDictionary.csv",
]

_DOLLAR_AMOUNT_PATTERN = re.compile(r"\$[\d,.]+\s*(?:billion|million|thousand|bn|mn|b|m)?", re.I)
_SHARE_COUNT_PATTERN = re.compile(r"[\d,.]+\s*(?:shares|million shares|billion shares|m shares)", re.I)
_TIMEFRAME_PATTERN = re.compile(
    r"\b(?:q[1-4]|quarter|quarters|month|months|year|years|through\s+20\d{2}|over the next|during)\b",
    re.I,
)
_FUNDING_PATTERN = re.compile(
    r"\b(?:free cash flow|cash on hand|debt|credit facility|operating cash|cash flow)\b",
    re.I,
)
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokenize_words(text: object) -> list[str]:
    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(str(text or ""))]


def compute_specificity(text: object) -> int:
    """Score the presence of concrete buyback details from 0 to 4."""

    text_value = str(text or "")
    score = 0
    score += int(bool(_DOLLAR_AMOUNT_PATTERN.search(text_value)))
    score += int(bool(_SHARE_COUNT_PATTERN.search(text_value)))
    score += int(bool(_TIMEFRAME_PATTERN.search(text_value)))
    score += int(bool(_FUNDING_PATTERN.search(text_value)))
    return score


def _resolve_lm_dictionary_path(path: str | Path | None) -> Path | None:
    if path is not None:
        resolved = Path(path)
        return resolved if resolved.exists() else None
    for candidate in DEFAULT_LM_DICTIONARY_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _read_lm_dictionary(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported Loughran-McDonald dictionary format: {path}")


def load_lm_dictionary(path: str | Path | None) -> tuple[set[str], set[str]]:
    """Load uncertainty and weak-modal word sets from the LM dictionary."""

    resolved_path = _resolve_lm_dictionary_path(path)
    if resolved_path is None:
        return DEFAULT_UNCERTAINTY_WORDS.copy(), DEFAULT_WEAK_MODAL_WORDS.copy()

    dictionary = _read_lm_dictionary(resolved_path)
    lower_map = {column.lower(): column for column in dictionary.columns}
    word_col = next(
        (
            lower_map[name]
            for name in ["word", "token", "term"]
            if name in lower_map
        ),
        None,
    )
    uncertainty_col = next(
        (
            lower_map[name]
            for name in ["uncertainty", "uncertain"]
            if name in lower_map
        ),
        None,
    )
    weak_modal_col = next(
        (
            lower_map[name]
            for name in ["weak modal", "weak_modal", "modal_weak", "modal"]
            if name in lower_map
        ),
        None,
    )
    if word_col is None or uncertainty_col is None or weak_modal_col is None:
        raise KeyError(
            "Could not find expected word / uncertainty / weak modal columns in the LM dictionary."
        )

    normalized_words = dictionary[word_col].astype(str).str.lower().str.strip()
    uncertainty_words = set(
        normalized_words.loc[pd.to_numeric(dictionary[uncertainty_col], errors="coerce").fillna(0) > 0]
    )
    weak_modal_words = set(
        normalized_words.loc[pd.to_numeric(dictionary[weak_modal_col], errors="coerce").fillna(0) > 0]
    )
    return uncertainty_words, weak_modal_words


def compute_hedge_density(text: object, lm_dict_path: str | Path | None = None) -> float:
    """Count uncertainty and weak-modal words normalized by total words."""

    words = _tokenize_words(text)
    if not words:
        return 0.0

    uncertainty_words, weak_modal_words = load_lm_dictionary(lm_dict_path)
    hedge_count = sum(word in uncertainty_words or word in weak_modal_words for word in words)
    return float(hedge_count / len(words))


def _sentence_chunks(text: object) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _count_syllables(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    if not normalized:
        return 0
    try:
        from textstat import textstat

        syllables = int(textstat.syllable_count(normalized))
        if syllables > 0:
            return syllables
    except Exception:  # pragma: no cover - fallback for minimal environments
        pass
    vowel_groups = re.findall(r"[aeiouy]+", normalized)
    syllables = len(vowel_groups)
    if normalized.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def compute_modified_fog(
    text: object,
    exclusion_set: Sequence[str] | None = None,
) -> float:
    """Compute a modified Gunning FOG index with finance-vocabulary exclusions."""

    sentences = _sentence_chunks(text)
    words = _tokenize_words(text)
    if not sentences or not words:
        return 0.0

    exclusions_source = FINANCIAL_VOCAB_EXCLUSIONS if exclusion_set is None else exclusion_set
    exclusions = {word.lower() for word in exclusions_source}
    complex_words = [
        word for word in words if _count_syllables(word) >= 3 and word.lower() not in exclusions
    ]
    words_per_sentence = len(words) / max(len(sentences), 1)
    complex_word_share = len(complex_words) / max(len(words), 1)
    return float(0.4 * (words_per_sentence + 100 * complex_word_share))


def _resolve_embedding_device(device: str | None = None) -> str:
    requested = device or "cuda"
    try:
        import torch
    except ImportError:  # pragma: no cover - runtime-only fallback
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@lru_cache(maxsize=2)
def _load_bge_model(model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_resolve_embedding_device(device))


def compute_qa_relevance(
    question: object,
    response: object,
    model=None,
    *,
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda",
) -> float:
    """Compute cosine similarity between an analyst question and executive response."""

    question_text = str(question or "").strip()
    response_text = str(response or "").strip()
    if not question_text or not response_text:
        return float("nan")

    encoder = model or _load_bge_model(model_name=model_name, device=device)
    embeddings = encoder.encode(
        [question_text, response_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def _safe_zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size <= 1 or np.nanstd(array) == 0:
        return np.zeros_like(array, dtype=float)
    return np.nan_to_num(zscore(array, nan_policy="omit"), nan=0.0)


def compute_clarity_composite(
    specificity,
    hedge_density,
    fog,
    qa_relevance,
):
    """Z-score each component and average into a clarity composite."""

    specificity_array = np.asarray(specificity, dtype=float)
    hedge_array = np.asarray(hedge_density, dtype=float)
    fog_array = np.asarray(fog, dtype=float)
    relevance_array = np.asarray(qa_relevance, dtype=float)

    stacked = np.vstack(
        [
            _safe_zscore(specificity_array),
            -_safe_zscore(hedge_array),
            -_safe_zscore(fog_array),
            _safe_zscore(relevance_array),
        ]
    )
    composite = np.nanmean(stacked, axis=0)
    if composite.ndim == 0:
        return float(composite)
    return composite


def _bucket_series(values: pd.Series, labels: list[str]) -> pd.Series:
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

    bucket_labels = [labels[0], labels[-1]] if q == 2 else labels
    assigned = pd.qcut(ranks, q=q, labels=bucket_labels, duplicates="drop")
    result.loc[valid.index] = assigned.astype(str)
    return result


def bucket_clarity(
    clarity_series: pd.Series,
    method: str = "tercile",
    groupby: pd.Series | None = None,
) -> pd.Categorical:
    """Bucket clarity into Low / Medium / High terciles."""

    if method != "tercile":
        raise ValueError("Only tercile bucketing is currently supported.")

    labels = ["Low", "Medium", "High"]
    if groupby is None:
        bucketed = _bucket_series(clarity_series, labels)
    else:
        group_index = pd.Series(groupby, index=clarity_series.index)
        bucketed = clarity_series.groupby(group_index).transform(
            lambda group: _bucket_series(group, labels)
        )
    return pd.Categorical(bucketed, categories=labels, ordered=True)
