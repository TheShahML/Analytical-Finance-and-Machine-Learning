"""Reusable loaders for the raw earnings-call transcript dataset."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from src.config.schemas import EXPECTED_TRANSCRIPT_COLUMNS
from src.config.settings import (
    DATA_DIR,
    LEGACY_TRANSCRIPT_PATH,
    RAW_TRANSCRIPT_PATH,
    TRANSCRIPT_FILENAME_CANDIDATES,
)


DEFAULT_DATE_COLUMNS = [
    "transcriptcreationdate_utc",
    "mostimportantdateutc",
    "call_date",
    "actual_call_date",
    "fiscal_period_end",
    "report_date",
    "guidance_date",
]


def resolve_transcript_path(path: str | Path | None = None) -> Path:
    """Return the preferred transcript file path."""

    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Transcript file not found: {resolved}")
        return resolved

    candidates = [RAW_TRANSCRIPT_PATH, LEGACY_TRANSCRIPT_PATH]
    for filename in TRANSCRIPT_FILENAME_CANDIDATES:
        candidates.extend([DATA_DIR / "raw" / filename, DATA_DIR / filename])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No transcript file found in the canonical raw directory or the current legacy path."
    )


def infer_file_type(path: str | Path) -> str:
    """Infer the file type from the path suffix."""

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    raise ValueError(f"Unsupported transcript file type for path: {path}")


def standardize_column_name(name: str) -> str:
    """Normalize a raw column name into a consistent snake-case style."""

    normalized = name.strip().lower()
    normalized = re.sub(r"[^\w]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def standardize_transcript_columns(
    df: pd.DataFrame, rename_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Return a copy of the dataset with standardized column names."""

    standardized = df.copy()
    standardized.columns = [standardize_column_name(column) for column in standardized.columns]
    if rename_map:
        standardized = standardized.rename(columns=rename_map)
    return standardized


def coerce_date_columns(
    df: pd.DataFrame, date_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Coerce configured date columns to pandas datetime where present."""

    coerced = df.copy()
    for column in date_columns or DEFAULT_DATE_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_datetime(coerced[column], errors="coerce")
    return coerced


def get_available_columns(path: str | Path | None = None) -> list[str]:
    """Read only the transcript header and return the available columns."""

    transcript_path = resolve_transcript_path(path)
    file_type = infer_file_type(transcript_path)

    if file_type == "csv":
        header = pd.read_csv(transcript_path, nrows=0)
    elif file_type == "parquet":
        header = pd.read_parquet(transcript_path, columns=[])
    else:
        header = pd.read_pickle(transcript_path)
        header = header.iloc[0:0]

    return list(standardize_transcript_columns(header).columns)


def load_raw_transcripts(
    path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
    standardize_columns: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load the raw transcript dataset from CSV, Parquet, or Pickle.

    Parameters
    ----------
    path:
        Optional override for the transcript file location.
    columns:
        Optional subset of columns to load.
    nrows:
        Optional row limit for fast iteration.
    standardize_columns:
        Whether to normalize column names to a stable snake-case style.
    parse_dates:
        Whether to coerce known date columns after loading.
    """

    transcript_path = resolve_transcript_path(path)
    file_type = infer_file_type(transcript_path)

    if file_type == "csv":
        df = pd.read_csv(transcript_path, usecols=columns, nrows=nrows)
    elif file_type == "parquet":
        df = pd.read_parquet(transcript_path, columns=columns)
        if nrows is not None:
            df = df.head(nrows)
    else:
        df = pd.read_pickle(transcript_path)
        if columns is not None:
            df = df.loc[:, list(columns)]
        if nrows is not None:
            df = df.head(nrows)

    if standardize_columns:
        df = standardize_transcript_columns(df)
    if parse_dates:
        df = coerce_date_columns(df)
    return df


def load_transcripts(
    path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper around `load_raw_transcripts`."""

    return load_raw_transcripts(path=path, columns=columns, nrows=nrows)


def compare_observed_to_expected_schema(path: str | Path | None = None) -> dict[str, list[str]]:
    """Compare the observed raw header to the scaffold's expected schema."""

    observed = set(get_available_columns(path))
    expected = set(EXPECTED_TRANSCRIPT_COLUMNS)
    return {
        "missing_from_observed": sorted(expected - observed),
        "extra_in_observed": sorted(observed - expected),
    }
