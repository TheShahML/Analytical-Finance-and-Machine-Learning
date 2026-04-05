"""Helpers for identifying buyback-related calls and text passages."""

from __future__ import annotations

from collections.abc import Sequence
import re

import pandas as pd


DEFAULT_BUYBACK_KEYWORDS: list[str] = [
    "buyback",
    "buy back",
    "repurchase",
    "share repurchase",
    "repurchasing",
    "stock repurchase",
    "repurchase program",
    "repurchase plan",
]

DEFAULT_TEXT_COLUMNS: tuple[str, ...] = (
    "full_transcript_text",
    "componenttext",
    "text",
    "content",
    "headline",
)


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def build_buyback_pattern(keywords: Sequence[str] | None = None) -> re.Pattern[str]:
    """Compile a case-insensitive regex for buyback language."""

    terms = keywords or DEFAULT_BUYBACK_KEYWORDS
    escaped_terms = [re.escape(term.lower()) for term in terms]
    pattern = rf"(?<!\w)(?:{'|'.join(escaped_terms)})(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def identify_buyback_transcripts(
    transcripts_df: pd.DataFrame,
    *,
    text_column: str | None = None,
    keywords: Sequence[str] | None = None,
    return_mask: bool = False,
) -> pd.Series | pd.DataFrame:
    """Flag transcript rows that mention buyback-related language."""

    columns_to_check = [text_column] if text_column else list(DEFAULT_TEXT_COLUMNS)
    available_columns = [column for column in columns_to_check if column in transcripts_df.columns]
    if not available_columns:
        raise KeyError(
            "Expected at least one transcript text column from "
            f"{list(columns_to_check)} but none were present."
        )

    pattern = build_buyback_pattern(keywords)
    normalized = (
        transcripts_df[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(_normalize_text)
    )
    mask = normalized.str.contains(pattern, na=False)
    return mask if return_mask else transcripts_df.loc[mask].copy()


def extract_buyback_sentences(
    text: object,
    keywords: Sequence[str] | None = None,
) -> list[str]:
    """Extract sentences that contain buyback-related language."""

    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []

    sentence_candidates = re.split(r"(?<=[.!?])\s+|\n+", normalized_text)
    pattern = build_buyback_pattern(keywords)
    sentences = [sentence.strip() for sentence in sentence_candidates if pattern.search(sentence)]
    return [sentence for sentence in sentences if sentence]


def match_buyback_events_to_transcripts(
    key_dev_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    window_days: int = 2,
    *,
    company_id_col: str = "companyid",
    transcript_date_col: str = "call_date",
    event_date_col: str = "event_date",
    event_type_col: str = "event_type",
    transcript_id_col: str = "transcriptid",
) -> pd.DataFrame:
    """Match buyback key-development events to nearby earnings-call transcripts."""

    required_key_dev = {company_id_col, event_date_col}
    required_transcripts = {company_id_col, transcript_date_col}
    missing_key_dev = required_key_dev - set(key_dev_df.columns)
    missing_transcripts = required_transcripts - set(transcripts_df.columns)
    if missing_key_dev:
        raise KeyError(f"Missing required key development columns: {sorted(missing_key_dev)}")
    if missing_transcripts:
        raise KeyError(f"Missing required transcript columns: {sorted(missing_transcripts)}")

    events = key_dev_df.copy()
    transcripts = transcripts_df.copy()
    events[event_date_col] = pd.to_datetime(events[event_date_col], errors="coerce")
    transcripts[transcript_date_col] = pd.to_datetime(
        transcripts[transcript_date_col], errors="coerce"
    )

    merged = events.merge(
        transcripts,
        on=company_id_col,
        how="left",
        suffixes=("_event", "_transcript"),
    )
    merged = merged.dropna(subset=[event_date_col, transcript_date_col]).copy()
    merged["date_distance_days"] = (
        merged[transcript_date_col] - merged[event_date_col]
    ).dt.days.abs()
    matched = merged.loc[merged["date_distance_days"] <= window_days].copy()

    sort_columns = ["date_distance_days"]
    if transcript_id_col in matched.columns:
        sort_columns.append(transcript_id_col)

    matched = matched.sort_values(sort_columns).drop_duplicates(
        subset=[company_id_col, event_date_col, event_type_col]
        if event_type_col in matched.columns
        else [company_id_col, event_date_col],
        keep="first",
    )
    return matched.reset_index(drop=True)
