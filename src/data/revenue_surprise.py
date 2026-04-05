"""Revenue-surprise utilities for buyback event analyses."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _find_first_available_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the expected columns were present: {list(candidates)}")


def compute_ibes_revenue_surprise(ibes_df: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue surprise as (actual - consensus) / consensus."""

    actual_col = _find_first_available_column(
        ibes_df,
        ["actual_revenue", "actual", "revenue_actual", "sales_actual"],
    )
    consensus_col = _find_first_available_column(
        ibes_df,
        ["consensus_revenue", "consensus", "mean_estimate", "revenue_estimate", "sales_estimate"],
    )

    result = ibes_df.copy()
    actual = pd.to_numeric(result[actual_col], errors="coerce")
    consensus = pd.to_numeric(result[consensus_col], errors="coerce")
    result["ibes_revenue_surprise"] = np.where(
        consensus.ne(0),
        (actual - consensus) / consensus,
        np.nan,
    )
    return result


def compute_trend_revenue_surprise(
    compustat_df: pd.DataFrame,
    lookback_quarters: int = 8,
) -> pd.DataFrame:
    """Estimate trend-based revenue surprise from recent firm revenue growth."""

    revenue_col = _find_first_available_column(
        compustat_df,
        ["revenue", "saleq", "sales", "actual_revenue"],
    )
    firm_col = _find_first_available_column(compustat_df, ["gvkey", "companyid", "permno", "ticker"])
    period_col = _find_first_available_column(
        compustat_df,
        ["datadate", "fiscal_period_end", "call_date", "report_date"],
    )

    result = compustat_df.copy()
    result[period_col] = pd.to_datetime(result[period_col], errors="coerce")
    result = result.sort_values([firm_col, period_col]).copy()

    revenue = pd.to_numeric(result[revenue_col], errors="coerce")
    result["_revenue_growth"] = revenue.groupby(result[firm_col]).pct_change()
    result["_expected_growth"] = (
        result.groupby(firm_col)["_revenue_growth"]
        .transform(lambda series: series.shift(1).rolling(lookback_quarters, min_periods=2).mean())
    )
    result["trend_revenue_surprise"] = result["_revenue_growth"] - result["_expected_growth"]
    return result.drop(columns=["_revenue_growth", "_expected_growth"])


def merge_revenue_surprise(
    transcripts_df: pd.DataFrame,
    ibes_surprise: pd.DataFrame,
    trend_surprise: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transcript observations with I/B/E/S and fallback trend surprises."""

    transcript_key = _find_first_available_column(
        transcripts_df,
        ["transcriptid", "companyid", "gvkey", "permno", "ticker"],
    )
    ibes_key = _find_first_available_column(ibes_surprise, [transcript_key, "companyid", "gvkey", "permno", "ticker"])
    trend_key = _find_first_available_column(
        trend_surprise,
        [transcript_key, "companyid", "gvkey", "permno", "ticker"],
    )

    merged = transcripts_df.copy()
    ibes_cols = [column for column in [ibes_key, "ibes_revenue_surprise"] if column in ibes_surprise.columns]
    trend_cols = [
        column for column in [trend_key, "trend_revenue_surprise"] if column in trend_surprise.columns
    ]

    merged = merged.merge(
        ibes_surprise.loc[:, ibes_cols].rename(columns={ibes_key: transcript_key}),
        on=transcript_key,
        how="left",
    )
    merged = merged.merge(
        trend_surprise.loc[:, trend_cols].rename(columns={trend_key: transcript_key}),
        on=transcript_key,
        how="left",
    )
    merged["revenue_surprise"] = merged["ibes_revenue_surprise"].fillna(
        merged["trend_revenue_surprise"]
    )
    return merged


def bucket_revenue(
    surprise_series: pd.Series,
    lower_quantile: float = 1 / 3,
    upper_quantile: float = 2 / 3,
) -> pd.Categorical:
    """Bucket revenue surprise into Below / In Line / Above expectations."""

    numeric = pd.to_numeric(surprise_series, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(pd.NA, index=numeric.index, dtype="object")

    if valid.empty:
        return pd.Categorical(result, categories=["Below", "In Line", "Above"], ordered=True)

    lower = valid.quantile(lower_quantile)
    upper = valid.quantile(upper_quantile)
    result.loc[numeric <= lower] = "Below"
    result.loc[(numeric > lower) & (numeric < upper)] = "In Line"
    result.loc[numeric >= upper] = "Above"
    return pd.Categorical(result, categories=["Below", "In Line", "Above"], ordered=True)
