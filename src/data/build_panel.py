"""Lightweight transcript-level outputs for validation and early analysis."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def flag_usable_transcripts(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    date_column: str = "call_date",
    transcript_id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Flag transcript rows that are usable for early EDA and downstream work."""

    required = [text_column, date_column, transcript_id_column]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for transcript usability checks: {missing_columns}")

    flagged = df.copy()
    reasons = pd.DataFrame(index=flagged.index)
    reasons["missing_transcript_id"] = flagged[transcript_id_column].isna()
    reasons["missing_call_date"] = pd.to_datetime(flagged[date_column], errors="coerce").isna()
    reasons["missing_or_blank_text"] = (
        flagged[text_column].isna() | flagged[text_column].astype(str).str.strip().eq("")
    )

    flagged["is_usable"] = ~reasons.any(axis=1)
    flagged["exclusion_reason"] = reasons.apply(
        lambda row: "; ".join(column for column, is_true in row.items() if is_true),
        axis=1,
    )
    return flagged


def summarize_usable_transcripts(flagged_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return summary tables for usable versus unusable observations."""

    if "is_usable" not in flagged_df.columns or "exclusion_reason" not in flagged_df.columns:
        raise KeyError("Expected `is_usable` and `exclusion_reason` columns in flagged dataset.")

    overall = (
        flagged_df["is_usable"]
        .value_counts(dropna=False)
        .rename_axis("is_usable")
        .reset_index(name="row_count")
    )
    overall["share"] = overall["row_count"] / max(len(flagged_df), 1)

    reasons = flagged_df.loc[~flagged_df["is_usable"], "exclusion_reason"]
    reason_counts = reasons.astype(str).str.strip()
    reason_counts = reason_counts[reason_counts != ""]
    if not reason_counts.empty:
        exploded = reason_counts.str.split("; ").explode()
        reason_summary = (
            exploded.value_counts().rename_axis("exclusion_reason").reset_index(name="row_count")
        )
    else:
        reason_summary = pd.DataFrame(columns=["exclusion_reason", "row_count"])

    return {"overall": overall, "reasons": reason_summary}


def filter_usable_transcripts(
    df: pd.DataFrame,
    text_column: str = "full_transcript_text",
    date_column: str = "call_date",
    transcript_id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Return only transcript rows usable for early analysis."""

    flagged = flag_usable_transcripts(
        df,
        text_column=text_column,
        date_column=date_column,
        transcript_id_column=transcript_id_column,
    )
    return flagged.loc[flagged["is_usable"]].copy()


def build_transcript_event_panel(
    df: pd.DataFrame,
    event_date_column: str = "actual_call_date",
    fallback_date_column: str = "call_date",
    key_columns: Sequence[str] = (
        "transcriptid",
        "companyid",
        "companyname",
        "ticker",
        "permno",
        "gvkey",
        "ibes_ticker",
    ),
) -> pd.DataFrame:
    """Build a lightweight transcript-level output for later notebooks."""

    panel = df.copy()
    if event_date_column in panel.columns:
        event_date = pd.to_datetime(panel[event_date_column], errors="coerce")
    else:
        event_date = pd.Series(pd.NaT, index=panel.index)

    if fallback_date_column in panel.columns:
        fallback_date = pd.to_datetime(panel[fallback_date_column], errors="coerce")
        event_date = event_date.fillna(fallback_date)

    panel["event_date"] = event_date
    selected_columns = [column for column in key_columns if column in panel.columns]
    extra_columns = [
        column
        for column in ["call_date", "actual_call_date", "event_date", "is_usable", "exclusion_reason"]
        if column in panel.columns
    ]
    return panel.loc[:, selected_columns + extra_columns].copy()
