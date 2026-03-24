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
    "prior_guidance_revenue",
    "guidance_low",
    "guidance_high",
    "guidance_date",
    "guidance_surprise",
    "guidance_surprise_pct",
    "forward_guidance_revenue",
    "forward_guidance_period",
    "ibes_ticker",
]

REQUIRED_TRANSCRIPT_COLUMNS = [
    "transcriptid",
    "companyname",
    "ticker",
    "full_transcript_text",
    "call_date",
]
