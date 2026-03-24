"""Cleaning helpers for transcript extraction issues and shared dataset construction."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a helpful error when required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def summarize_duplicate_event_groups(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("ticker", "call_date"),
) -> dict[str, pd.DataFrame]:
    """Summarize duplicate transcript groups for likely event-level revisions."""

    _require_columns(df, group_cols)
    counts = df.groupby(list(group_cols), dropna=False).size().reset_index(name="row_count")
    duplicate_groups = counts.loc[counts["row_count"] > 1].sort_values(
        "row_count", ascending=False
    )

    group_size = df.groupby(list(group_cols), dropna=False)[group_cols[0]].transform("size")
    rows_in_duplicate_groups = int((group_size > 1).sum())

    summary = pd.DataFrame(
        [
            {
                "total_rows": int(len(df)),
                "rows_in_duplicate_groups": rows_in_duplicate_groups,
                "rows_in_duplicate_groups_share": float(rows_in_duplicate_groups / max(len(df), 1)),
                "duplicate_groups": int(len(duplicate_groups)),
                "max_group_size": int(duplicate_groups["row_count"].max())
                if not duplicate_groups.empty
                else 1,
            }
        ]
    )
    return {"summary": summary, "duplicate_groups": duplicate_groups}


def sample_duplicate_event_examples(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("ticker", "call_date"),
    transcript_id_col: str = "transcriptid",
    text_col: str = "full_transcript_text",
    creation_col: str = "transcriptcreationdate_utc",
    transcript_length_col: str = "transcript_length",
    min_group_size: int = 3,
    sample_n: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return example duplicate groups for notebook inspection."""

    _require_columns(df, [*group_cols, transcript_id_col, text_col])
    group_summary = summarize_duplicate_event_groups(df, group_cols)["duplicate_groups"]
    eligible = group_summary.loc[group_summary["row_count"] >= min_group_size]
    if eligible.empty:
        return pd.DataFrame()

    selected = eligible.sample(n=min(sample_n, len(eligible)), random_state=random_state)
    examples = df.merge(selected[list(group_cols)], on=list(group_cols), how="inner").copy()

    text_series = examples[text_col].fillna("").astype(str)
    examples["text_len_observed"] = (
        pd.to_numeric(examples[transcript_length_col], errors="coerce")
        if transcript_length_col in examples.columns
        else text_series.str.len()
    )
    examples["text_preview"] = text_series.str.slice(0, 120)

    columns = [column for column in [*group_cols, transcript_id_col, creation_col, transcript_length_col] if column in examples.columns]
    columns += ["text_len_observed", "text_preview"]
    return examples.loc[:, columns].sort_values(list(group_cols) + ["text_len_observed"], ascending=[True] * len(group_cols) + [False])


def summarize_preperiod_leakage(
    df: pd.DataFrame,
    call_date_col: str = "call_date",
    creation_date_col: str = "transcriptcreationdate_utc",
    min_call_date: str = "2010-01-01",
    sample_n: int = 5,
) -> dict[str, pd.DataFrame]:
    """Summarize rows that fall before the intended call-date window."""

    _require_columns(df, [call_date_col, creation_date_col])
    call_dates = pd.to_datetime(df[call_date_col], errors="coerce")
    creation_dates = pd.to_datetime(df[creation_date_col], errors="coerce")
    cutoff = pd.Timestamp(min_call_date)

    preperiod = df.loc[call_dates < cutoff].copy()
    preperiod["call_year"] = call_dates.loc[preperiod.index].dt.year
    preperiod["creation_year"] = creation_dates.loc[preperiod.index].dt.year

    summary = pd.DataFrame(
        [
            {
                "cutoff_date": cutoff.date(),
                "preperiod_rows": int(len(preperiod)),
                "preperiod_share": float(len(preperiod) / max(len(df), 1)),
                "all_creation_dates_on_or_after_cutoff": bool(
                    preperiod["creation_year"].dropna().ge(cutoff.year).all()
                )
                if not preperiod.empty
                else True,
            }
        ]
    )

    call_year_counts = (
        preperiod["call_year"].value_counts().sort_index().rename_axis("call_year").reset_index(name="row_count")
        if not preperiod.empty
        else pd.DataFrame(columns=["call_year", "row_count"])
    )
    creation_year_counts = (
        preperiod["creation_year"]
        .value_counts()
        .sort_index()
        .rename_axis("creation_year")
        .reset_index(name="row_count")
        if not preperiod.empty
        else pd.DataFrame(columns=["creation_year", "row_count"])
    )

    sample_columns = [
        column
        for column in ["ticker", "companyname", call_date_col, creation_date_col]
        if column in preperiod.columns
    ]
    sample_rows = (
        preperiod.loc[:, sample_columns].head(sample_n).reset_index(drop=True)
        if not preperiod.empty
        else pd.DataFrame(columns=sample_columns)
    )

    return {
        "summary": summary,
        "call_year_counts": call_year_counts,
        "creation_year_counts": creation_year_counts,
        "sample_rows": sample_rows,
    }


def summarize_exact_zero_return_issue(
    df: pd.DataFrame,
    return_col: str = "close_to_open_return",
    close_col: str = "close_price_call_day",
    open_col: str = "open_price_next_day",
    price_threshold: float = 50.0,
    sample_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """Summarize exact-zero return rows that may reflect stale open prices."""

    _require_columns(df, [return_col, close_col, open_col])
    returns = pd.to_numeric(df[return_col], errors="coerce")
    close_prices = pd.to_numeric(df[close_col], errors="coerce")
    open_prices = pd.to_numeric(df[open_col], errors="coerce")

    zero_rows = df.loc[returns.eq(0)].copy()
    zero_close = close_prices.loc[zero_rows.index]
    zero_open = open_prices.loc[zero_rows.index]
    exact_match_mask = zero_close.eq(zero_open)

    summary = pd.DataFrame(
        [
            {
                "zero_return_rows": int(len(zero_rows)),
                "zero_return_share": float(len(zero_rows) / max(len(df), 1)),
                "exact_close_open_matches": int(exact_match_mask.sum()),
                "exact_match_share_within_zero_returns": float(exact_match_mask.mean())
                if len(zero_rows) > 0
                else 0.0,
                "high_price_zero_rows": int(zero_close.gt(price_threshold).sum()),
            }
        ]
    )

    sample_columns = [
        column
        for column in ["ticker", "companyname", "call_date", close_col, open_col]
        if column in zero_rows.columns
    ]
    expensive_samples = (
        zero_rows.loc[zero_close.gt(price_threshold), sample_columns]
        .drop_duplicates(subset=[column for column in ["ticker", "call_date"] if column in sample_columns])
        .head(sample_n)
        .reset_index(drop=True)
    )

    return {"summary": summary, "sample_rows": expensive_samples}


def _selection_length(
    df: pd.DataFrame,
    transcript_length_col: str = "transcript_length",
    text_col: str = "full_transcript_text",
) -> pd.Series:
    """Return a length metric for transcript selection during deduplication."""

    if transcript_length_col in df.columns:
        length = pd.to_numeric(df[transcript_length_col], errors="coerce")
        if length.notna().all():
            return length

        fallback = df[text_col].fillna("").astype(str).str.len() if text_col in df.columns else 0
        return length.fillna(fallback)

    if text_col in df.columns:
        return df[text_col].fillna("").astype(str).str.len()

    raise KeyError(
        "Need either a reported transcript length column or transcript text to deduplicate by length."
    )


def deduplicate_transcript_events(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("ticker", "call_date"),
    transcript_length_col: str = "transcript_length",
    text_col: str = "full_transcript_text",
) -> pd.DataFrame:
    """Deduplicate likely transcript revisions by keeping the longest version per event."""

    _require_columns(df, group_cols)
    deduped = df.copy()
    deduped["_selection_length"] = _selection_length(
        deduped,
        transcript_length_col=transcript_length_col,
        text_col=text_col,
    )
    deduped = deduped.sort_values("_selection_length", ascending=False)
    deduped = deduped.drop_duplicates(subset=list(group_cols), keep="first")
    return deduped.drop(columns="_selection_length")


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    call_date_col: str = "call_date",
    min_call_date: str = "2010-01-01",
    dedup_group_cols: Sequence[str] = ("ticker", "call_date"),
    transcript_length_col: str = "transcript_length",
    text_col: str = "full_transcript_text",
    close_col: str = "close_price_call_day",
    open_col: str = "open_price_next_day",
    remove_exact_zero_returns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the Task 01 cleaning rules and return a cleaned dataset plus a log."""

    _require_columns(df, [call_date_col, *dedup_group_cols])
    cleaned = df.copy()
    log_rows: list[dict[str, object]] = []

    before = len(cleaned)
    call_dates = pd.to_datetime(cleaned[call_date_col], errors="coerce")
    cleaned = cleaned.loc[call_dates >= pd.Timestamp(min_call_date)].copy()
    log_rows.append(
        {
            "step": "filter_preperiod_rows",
            "rows_before": before,
            "rows_after": len(cleaned),
            "rows_dropped": before - len(cleaned),
            "notes": f"Kept {call_date_col} >= {min_call_date}",
        }
    )

    before = len(cleaned)
    cleaned = deduplicate_transcript_events(
        cleaned,
        group_cols=dedup_group_cols,
        transcript_length_col=transcript_length_col,
        text_col=text_col,
    )
    log_rows.append(
        {
            "step": "deduplicate_event_revisions",
            "rows_before": before,
            "rows_after": len(cleaned),
            "rows_dropped": before - len(cleaned),
            "notes": f"Kept longest transcript per {tuple(dedup_group_cols)}",
        }
    )

    if remove_exact_zero_returns:
        _require_columns(cleaned, [close_col, open_col])
        before = len(cleaned)
        close_prices = pd.to_numeric(cleaned[close_col], errors="coerce")
        open_prices = pd.to_numeric(cleaned[open_col], errors="coerce")
        cleaned = cleaned.loc[~close_prices.eq(open_prices)].copy()
        log_rows.append(
            {
                "step": "remove_exact_zero_return_rows",
                "rows_before": before,
                "rows_after": len(cleaned),
                "rows_dropped": before - len(cleaned),
                "notes": f"Dropped rows where {close_col} == {open_col}",
            }
        )

    return cleaned.reset_index(drop=True), pd.DataFrame(log_rows)
