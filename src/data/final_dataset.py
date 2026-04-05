"""Helpers for working with the canonical cleaned `FINAL.csv` dataset."""

from __future__ import annotations

import re

import pandas as pd


FINAL_COLUMN_MAP: dict[str, str] = {
    "transcriptid": "Unique transcript identifier",
    "companyid": "Company identifier from the transcript source",
    "companyname": "Company name",
    "ticker": "Ticker symbol",
    "permno": "CRSP PERMNO identifier",
    "gvkey": "Compustat GVKEY identifier",
    "headline": "Transcript headline",
    "event_type": "Event label in the source data",
    "full_transcript_text": "Full transcript text for transcript-level NLP",
    "call_date": "Primary earnings-call event date",
    "actual_call_date": "Alternative event date field from source metadata",
    "transcriptcreationdate_utc": "Transcript creation date",
    "mostimportantdateutc": "Important date from source metadata",
    "transcript_length": "Reported transcript character length",
    "word_count": "Reported transcript word count",
    "close_price_call_day": "Closing price on the call day",
    "open_price_next_day": "Opening price on the next trading day",
    "close_to_open_return": "Close-to-next-open return",
    "ret_t-15": "Relative-day return, t-15",
    "ret_t-1": "Relative-day return, t-1",
    "ret_t0": "Relative-day return, t0",
    "ret_t1": "Relative-day return, t+1",
    "ret_t15": "Relative-day return, t+15",
    "fiscal_period_end": "Fiscal quarter end date",
    "report_date": "Report or earnings release date",
    "fiscal_year": "Fiscal year",
    "fiscal_quarter": "Fiscal quarter",
    "compustat_actual_revenue": "Actual revenue from Compustat",
    "actual_revenue": "Alias for Compustat actual revenue",
    "ibes_ticker": "I/B/E/S ticker",
    "ibes_anndats": "I/B/E/S announcement date",
    "ibes_announcement_date": "Alias for I/B/E/S announcement date",
    "ibes_mean_est_eps": "I/B/E/S mean EPS estimate",
    "ibes_actual_eps": "I/B/E/S actual EPS",
    "ibes_raw_surp_eps": "Raw EPS surprise from I/B/E/S",
    "ibes_sue_eps": "Standardized unexpected earnings from I/B/E/S",
}


def get_final_column_map(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a tidy description of the FINAL.csv column mapping."""

    records = []
    for column, meaning in FINAL_COLUMN_MAP.items():
        records.append(
            {
                "column": column,
                "meaning": meaning,
                "present": bool(df is None or column in df.columns),
            }
        )

    if df is not None:
        for column in df.columns:
            if column not in FINAL_COLUMN_MAP:
                records.append(
                    {
                        "column": column,
                        "meaning": "Unmapped/project-specific column",
                        "present": True,
                    }
                )
    return pd.DataFrame(records).sort_values("column").reset_index(drop=True)


def get_wide_return_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted `ret_t...` columns from the cleaned dataset."""

    columns = [column for column in df.columns if re.fullmatch(r"ret_t-?\d+", column)]
    return sorted(columns, key=lambda column: int(column.replace("ret_t", "")))


def summarize_final_dataset_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which research steps can run directly off `FINAL.csv`."""

    has_qa_fields = {"transcriptcomponenttypeid", "speakertypeid"}.issubset(df.columns)
    capabilities = [
        {
            "workflow_step": "Transcript-level buyback detection",
            "supported": "full_transcript_text" in df.columns,
            "notes": "Supported from transcript text.",
        },
        {
            "workflow_step": "Prepared remarks vs Q&A split",
            "supported": has_qa_fields,
            "notes": "Requires a separate component-level transcript export keyed by transcriptid.",
        },
        {
            "workflow_step": "Transcript-level FinBERT sentiment",
            "supported": "full_transcript_text" in df.columns,
            "notes": "Supported from full transcript text and buyback sentence extraction.",
        },
        {
            "workflow_step": "Buyback Q&A clarity composite",
            "supported": has_qa_fields,
            "notes": "Requires a separate component-level transcript export with speaker/component fields.",
        },
        {
            "workflow_step": "Revenue surprise from I/B/E/S revenue estimates",
            "supported": "consensus_revenue" in df.columns,
            "notes": "Not present; use trend-based revenue surprise fallback instead.",
        },
        {
            "workflow_step": "Trend-based revenue surprise",
            "supported": "actual_revenue" in df.columns or "compustat_actual_revenue" in df.columns,
            "notes": "Supported from Compustat actual revenue history.",
        },
        {
            "workflow_step": "Post-event return windows from wide returns",
            "supported": bool(get_wide_return_columns(df)),
            "notes": "Supported for direct t+1:t+3 and t+1:t+5 sums.",
        },
        {
            "workflow_step": "Full abnormal-return event study [-120, -20]",
            "supported": False,
            "notes": "WIP for FINAL.csv because only ret_t-15 to ret_t15 are available.",
        },
        {
            "workflow_step": "Image-spec event study using wide returns",
            "supported": bool(get_wide_return_columns(df)),
            "notes": "Supported with estimation window [-15, -3] and post-event windows [+1, +3] and [+1, +5].",
        },
    ]
    return pd.DataFrame(capabilities)
