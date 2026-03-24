"""Reusable validation and audit helpers for transcript datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd

from src.config.schemas import REQUIRED_TRANSCRIPT_COLUMNS


@dataclass(frozen=True)
class ValidationSummary:
    """Compact summary for quick validation checks."""

    row_count: int
    missing_required_columns: list[str]
    duplicate_transcript_ids: int
    missing_text_rows: int
    missing_call_date_rows: int


def _missing_text_mask(df: pd.DataFrame, text_column: str) -> pd.Series:
    """Return a boolean mask for missing or blank transcript text."""

    if text_column not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    text = df[text_column]
    return text.isna() | text.astype(str).str.strip().eq("")


def validate_transcript_schema(
    columns: Iterable[str], required_columns: Sequence[str] | None = None
) -> list[str]:
    """Return required columns missing from the dataset."""

    required = list(required_columns or REQUIRED_TRANSCRIPT_COLUMNS)
    observed = set(columns)
    return [column for column in required if column not in observed]


def check_required_columns(
    df: pd.DataFrame, required_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a table showing whether each required column is present."""

    required = list(required_columns or REQUIRED_TRANSCRIPT_COLUMNS)
    observed = set(df.columns)
    rows = [{"column": column, "present": column in observed} for column in required]
    return pd.DataFrame(rows)


def summarize_schema(df: pd.DataFrame, sample_values: int = 2) -> pd.DataFrame:
    """Summarize column names, dtypes, non-null counts, and sample values."""

    rows: list[dict[str, object]] = []
    for column in df.columns:
        non_null = df[column].dropna()
        example_values = [str(value)[:120] for value in non_null.head(sample_values).tolist()]
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "missing_count": int(df[column].isna().sum()),
                "missing_rate": float(df[column].isna().mean()),
                "sample_values": " | ".join(example_values),
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness counts and rates for all columns."""

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
            "missing_rate": [float(df[column].isna().mean()) for column in df.columns],
            "non_null_count": [int(df[column].notna().sum()) for column in df.columns],
        }
    )
    return summary.sort_values(["missing_rate", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def find_duplicate_rows(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    """Return duplicate rows for a given subset of columns."""

    return df.loc[df.duplicated(subset=list(subset) if subset else None, keep=False)].copy()


def summarize_date_coverage(df: pd.DataFrame, date_col: str) -> dict[str, pd.DataFrame]:
    """Return date coverage summary tables for a configured date column."""

    if date_col not in df.columns:
        raise KeyError(f"Date column `{date_col}` not found in dataset.")

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    year_counts = (
        parsed.dt.year.value_counts(dropna=True).sort_index().rename_axis("year").reset_index(name="transcript_count")
    )

    summary = pd.DataFrame(
        [
            {
                "date_column": date_col,
                "non_missing_count": int(parsed.notna().sum()),
                "missing_count": int(parsed.isna().sum()),
                "min_date": parsed.min(),
                "max_date": parsed.max(),
                "distinct_years": int(parsed.dt.year.nunique(dropna=True)),
            }
        ]
    )
    return {"summary": summary, "by_year": year_counts}


def summarize_firm_coverage(
    df: pd.DataFrame, firm_id_col: str, top_n: int = 20
) -> dict[str, pd.DataFrame]:
    """Return firm-level coverage summary tables."""

    if firm_id_col not in df.columns:
        raise KeyError(f"Firm identifier column `{firm_id_col}` not found in dataset.")

    firm_series = df[firm_id_col]
    non_missing = firm_series.dropna()
    counts = (
        non_missing.value_counts().rename_axis(firm_id_col).reset_index(name="transcript_count")
    )

    summary = pd.DataFrame(
        [
            {
                "firm_id_column": firm_id_col,
                "non_missing_count": int(firm_series.notna().sum()),
                "missing_count": int(firm_series.isna().sum()),
                "unique_firms": int(non_missing.nunique()),
            }
        ]
    )
    return {"summary": summary, "top_firms": counts.head(top_n)}


def summarize_identifier_match_rate(
    df: pd.DataFrame, id_cols: Sequence[str]
) -> pd.DataFrame:
    """Return non-missing rates for configured identifier columns."""

    rows: list[dict[str, object]] = []
    row_count = max(len(df), 1)
    for column in id_cols:
        if column not in df.columns:
            rows.append(
                {
                    "identifier_column": column,
                    "present": False,
                    "non_missing_count": 0,
                    "match_rate": 0.0,
                    "unique_non_missing": 0,
                }
            )
            continue

        series = df[column]
        non_missing = series.notna().sum()
        rows.append(
            {
                "identifier_column": column,
                "present": True,
                "non_missing_count": int(non_missing),
                "match_rate": float(non_missing / row_count),
                "unique_non_missing": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def summarize_text_length(
    df: pd.DataFrame,
    text_col: str,
    reported_character_count_col: str | None = "transcript_length",
    reported_word_count_col: str | None = "word_count",
    compute_from_text: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return text-length summary tables and row-level metrics.

    By default, this function prefers existing reported length columns because
    deriving word counts from a very large raw transcript column can be memory
    intensive. Set `compute_from_text=True` only when text-derived metrics are
    explicitly needed.
    """

    if text_col not in df.columns:
        raise KeyError(f"Text column `{text_col}` not found in dataset.")

    metrics = pd.DataFrame(index=df.index)

    if reported_character_count_col and reported_character_count_col in df.columns:
        metrics["reported_character_count"] = pd.to_numeric(
            df[reported_character_count_col], errors="coerce"
        )

    if reported_word_count_col and reported_word_count_col in df.columns:
        reported = pd.to_numeric(df[reported_word_count_col], errors="coerce")
        metrics["reported_word_count"] = reported

    if compute_from_text:
        text = df[text_col].fillna("").astype(str)
        metrics["computed_character_count"] = text.str.len()
        metrics["computed_word_count"] = text.str.split().str.len()
        if "reported_word_count" in metrics.columns:
            metrics["word_count_difference"] = (
                metrics["reported_word_count"] - metrics["computed_word_count"]
            )

    summary_rows = []
    for column in metrics.columns:
        non_null = metrics[column].dropna()
        if non_null.empty:
            continue
        summary_rows.append(
            {
                "metric": column,
                "count": int(non_null.shape[0]),
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "min": float(non_null.min()),
                "p25": float(non_null.quantile(0.25)),
                "p75": float(non_null.quantile(0.75)),
                "max": float(non_null.max()),
            }
        )
    return {"summary": pd.DataFrame(summary_rows), "row_metrics": metrics}


def build_validation_summary(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
    id_column: str = "transcriptid",
    text_column: str = "full_transcript_text",
    date_column: str = "call_date",
) -> ValidationSummary:
    """Construct a compact validation summary for a transcript dataset."""

    missing_required = validate_transcript_schema(df.columns, required_columns)

    duplicate_ids = 0
    if id_column in df.columns:
        duplicate_ids = int(df[id_column].duplicated().sum())

    missing_text = int(_missing_text_mask(df, text_column).sum())

    missing_dates = 0
    if date_column in df.columns:
        missing_dates = int(pd.to_datetime(df[date_column], errors="coerce").isna().sum())

    return ValidationSummary(
        row_count=len(df),
        missing_required_columns=missing_required,
        duplicate_transcript_ids=duplicate_ids,
        missing_text_rows=missing_text,
        missing_call_date_rows=missing_dates,
    )


def validation_summary_to_frame(summary: ValidationSummary) -> pd.DataFrame:
    """Convert a validation summary to a single-row DataFrame."""

    return pd.DataFrame([asdict(summary)])


def build_validation_report(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
    date_col: str = "call_date",
    firm_id_col: str = "companyname",
    text_col: str = "full_transcript_text",
    transcript_id_col: str = "transcriptid",
    identifier_columns: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a reusable validation report for the Task 1 notebook.

    TODO:
    - adjust configured identifier columns if the raw schema changes
    - add near-duplicate text checks if the corpus requires them
    """

    identifier_columns = list(
        identifier_columns
        or ["transcriptid", "companyid", "ticker", "permno", "gvkey", "ibes_ticker"]
    )
    duplicate_subset = [transcript_id_col] if transcript_id_col in df.columns else None

    report: dict[str, pd.DataFrame] = {
        "required_columns": check_required_columns(df, required_columns),
        "schema_summary": summarize_schema(df),
        "missingness_summary": summarize_missingness(df),
        "validation_summary": validation_summary_to_frame(
            build_validation_summary(
                df,
                required_columns=required_columns,
                id_column=transcript_id_col,
                text_column=text_col,
                date_column=date_col,
            )
        ),
        "duplicate_rows": find_duplicate_rows(df, subset=duplicate_subset),
        "identifier_match_rate": summarize_identifier_match_rate(df, identifier_columns),
    }

    if date_col in df.columns:
        report.update(
            {
                "date_summary": summarize_date_coverage(df, date_col)["summary"],
                "date_by_year": summarize_date_coverage(df, date_col)["by_year"],
            }
        )

    if firm_id_col in df.columns:
        firm_report = summarize_firm_coverage(df, firm_id_col)
        report["firm_summary"] = firm_report["summary"]
        report["top_firms"] = firm_report["top_firms"]

    if text_col in df.columns:
        text_report = summarize_text_length(df, text_col, compute_from_text=False)
        report["text_length_summary"] = text_report["summary"]

    return report
