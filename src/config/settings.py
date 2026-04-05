"""Project-level paths and default configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TRANSCRIPT_FILENAME = "FINAL.csv"
TRANSCRIPT_FILENAME_CANDIDATES = [
    DEFAULT_TRANSCRIPT_FILENAME,
    "final.csv",
    "earnings_calls_full_2010_onward_with_revenue.csv",
    "earnings_calls_full_5year_with_revenue.csv",
]
CANONICAL_TRANSCRIPT_PATH = DATA_DIR / DEFAULT_TRANSCRIPT_FILENAME
RAW_TRANSCRIPT_PATH = DATA_DIR / "raw" / DEFAULT_TRANSCRIPT_FILENAME
LEGACY_TRANSCRIPT_PATH = DATA_DIR / "earnings_calls_full_2010_onward_with_revenue.csv"
TRANSCRIPT_COMPONENT_FILENAME_CANDIDATES = [
    "wrds_transcript_components.csv",
    "true_transcript_components.csv",
    "FINAL_COMPONENTS.csv",
    "final_components.csv",
    "transcript_components.csv",
    "transcript_component_rows.csv",
    "qa_components.csv",
]


@dataclass(frozen=True)
class ProjectPaths:
    """Container for frequently used project directories."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = DATA_DIR / "raw"
    interim_dir: Path = DATA_DIR / "interim"
    processed_dir: Path = DATA_DIR / "processed"
    external_dir: Path = DATA_DIR / "external"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"
    docs_dir: Path = PROJECT_ROOT / "docs"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
