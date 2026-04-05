"""Loaders for component-level transcript rows used in Q&A speaker splitting.

These files are separate from `data/FINAL.csv`. The cleaned transcript-level
dataset remains canonical, while component-level exports enable prepared
remarks vs Q&A and analyst vs executive speaker splits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config.settings import (
    DATA_DIR,
    TRANSCRIPT_COMPONENT_FILENAME_CANDIDATES,
)
from src.data.load_transcripts import coerce_date_columns, standardize_transcript_columns


DEFAULT_COMPONENT_COLUMN_ALIASES = {
    "component_text": "componenttext",
    "speaker_type_id": "speakertypeid",
    "transcript_component_type_id": "transcriptcomponenttypeid",
    "speaker_name": "speakername",
    "component_order": "componentorder",
    "sequence_number": "componentorder",
}


def resolve_transcript_component_path(path: str | Path | None = None) -> Path:
    """Return the preferred transcript-component file path."""

    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Transcript component file not found: {resolved}")
        return resolved

    env_path = os.getenv("TRANSCRIPT_COMPONENTS_PATH")
    if env_path:
        resolved = Path(env_path)
        if resolved.exists():
            return resolved

    candidates: list[Path] = []
    for filename in TRANSCRIPT_COMPONENT_FILENAME_CANDIDATES:
        candidates.extend(
            [
                DATA_DIR / "interim" / filename,
                DATA_DIR / "raw" / filename,
                DATA_DIR / filename,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No transcript component file found. Add a component-level export such as "
        "`data/interim/transcript_components.csv` to enable the Q&A speaker split."
    )


def load_transcript_components(
    path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
    transcript_ids: Sequence[int | str] | None = None,
    chunksize: int = 250_000,
    standardize_columns: bool = True,
    parse_dates: bool = True,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a component-level transcript export for Q&A splitting."""

    component_path = resolve_transcript_component_path(path)
    suffix = component_path.suffix.lower()
    use_columns = list(columns) if columns is not None else None

    transcript_id_filter: set[int] | set[str] | None = None
    if transcript_ids is not None:
        transcript_id_filter = {
            _normalize_transcript_id(transcript_id)
            for transcript_id in transcript_ids
            if pd.notna(transcript_id)
        }
        if transcript_id_filter and use_columns is not None and "transcriptid" not in use_columns:
            use_columns = [*use_columns, "transcriptid"]

    if suffix == ".csv":
        if transcript_id_filter:
            chunks: list[pd.DataFrame] = []
            rows_collected = 0
            for chunk in pd.read_csv(component_path, usecols=use_columns, chunksize=chunksize):
                if "transcriptid" not in chunk.columns:
                    raise KeyError("Expected `transcriptid` column when filtering component rows.")
                normalized_ids = chunk["transcriptid"].map(_normalize_transcript_id)
                filtered = chunk.loc[normalized_ids.isin(transcript_id_filter)].copy()
                if filtered.empty:
                    continue

                if nrows is not None:
                    remaining = nrows - rows_collected
                    if remaining <= 0:
                        break
                    filtered = filtered.head(remaining)

                chunks.append(filtered)
                rows_collected += len(filtered)
                if nrows is not None and rows_collected >= nrows:
                    break

            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=use_columns)
        else:
            df = pd.read_csv(component_path, usecols=use_columns, nrows=nrows)
    elif suffix == ".parquet":
        df = pd.read_parquet(component_path, columns=use_columns)
        if transcript_id_filter:
            if "transcriptid" not in df.columns:
                raise KeyError("Expected `transcriptid` column when filtering component rows.")
            normalized_ids = df["transcriptid"].map(_normalize_transcript_id)
            df = df.loc[normalized_ids.isin(transcript_id_filter)].copy()
        if nrows is not None:
            df = df.head(nrows)
    else:
        raise ValueError(f"Unsupported transcript component file type: {component_path}")

    if standardize_columns:
        combined_rename_map = DEFAULT_COMPONENT_COLUMN_ALIASES | (rename_map or {})
        df = standardize_transcript_columns(df, rename_map=combined_rename_map)
    if parse_dates:
        df = coerce_date_columns(df)
    return df


def _normalize_transcript_id(value: int | str | float | object) -> int | str:
    """Normalize transcript identifiers across CSV/Parquet type differences."""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    try:
        numeric_value = pd.to_numeric(value, errors="raise")
    except (TypeError, ValueError):
        return str(value)

    if pd.isna(numeric_value):
        return ""
    return int(numeric_value)


def component_data_supports_qa_split(df: pd.DataFrame) -> bool:
    """Return whether a component-level export can support the Q&A split."""

    required = {"transcriptid", "transcriptcomponenttypeid", "speakertypeid"}
    return required.issubset(df.columns)


def merge_transcript_components(
    transcripts_df: pd.DataFrame,
    components_df: pd.DataFrame,
    transcript_id_col: str = "transcriptid",
) -> pd.DataFrame:
    """Restrict component rows to transcript IDs present in the canonical transcript panel."""

    if transcript_id_col not in transcripts_df.columns or transcript_id_col not in components_df.columns:
        raise KeyError(f"Expected transcript ID column `{transcript_id_col}` in both datasets.")

    valid_ids = transcripts_df[transcript_id_col].dropna().unique()
    return components_df.loc[components_df[transcript_id_col].isin(valid_ids)].copy()
