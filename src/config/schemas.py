"""Observed and required schema definitions for transcript workflows."""

from __future__ import annotations


EXPECTED_TRANSCRIPT_COLUMNS = [
    "transcriptid",
    "companyid",
    "headline",
    "transcriptcreationdate_utc",
    "mostimportantdateutc",
    "companyname",
    "ticker",
    "event_type",
    "full_transcript_text",
    "call_date",
    "permno",
    "actual_call_date",
    "close_price_call_day",
    "open_price_next_day",
    "close_to_open_return",
    "transcript_length",
    "word_count",
    "gvkey",
    "fiscal_period_end",
    "report_date",
    "fiscal_year",
    "fiscal_quarter",
    "actual_revenue",
    "ibes_ticker",
    "ibes_announcement_date",
    "ibes_mean_est_eps",
    "ibes_actual_eps",
    "ibes_raw_surp_eps",
    "ibes_sue_eps",
]

REQUIRED_TRANSCRIPT_COLUMNS = [
    "transcriptid",
    "companyname",
    "ticker",
    "full_transcript_text",
    "call_date",
]
