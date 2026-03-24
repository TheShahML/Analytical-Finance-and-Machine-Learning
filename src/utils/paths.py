"""Centralized project-relative paths for notebooks and reusable modules."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import (
    DATA_DIR,
    DEFAULT_TRANSCRIPT_FILENAME,
    ProjectPaths,
    TRANSCRIPT_FILENAME_CANDIDATES,
)


PATHS = ProjectPaths()

PROJECT_ROOT = PATHS.project_root
DATA_DIR = PATHS.data_dir
RAW_DATA_DIR = PATHS.raw_dir
INTERIM_DATA_DIR = PATHS.interim_dir
PROCESSED_DATA_DIR = PATHS.processed_dir
EXTERNAL_DATA_DIR = PATHS.external_dir
DOCS_DIR = PATHS.docs_dir
OUTPUTS_DIR = PATHS.outputs_dir
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a `Path`."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_project_dirs() -> None:
    """Ensure the standard project output and data directories exist."""

    for directory in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        ensure_dir(directory)


def transcript_raw_path(filename: str = DEFAULT_TRANSCRIPT_FILENAME) -> Path:
    """Return the preferred raw transcript path.

    The project still supports the legacy top-level `data/` location while the
    raw file has not yet been moved into `data/raw/`.
    """

    candidate_filenames = [filename]
    candidate_filenames.extend(
        [candidate for candidate in TRANSCRIPT_FILENAME_CANDIDATES if candidate != filename]
    )

    for candidate_filename in candidate_filenames:
        canonical = RAW_DATA_DIR / candidate_filename
        legacy_candidate = DATA_DIR / candidate_filename
        if canonical.exists():
            return canonical

        if legacy_candidate.exists():
            return legacy_candidate

    return canonical


def raw_data_path(filename: str) -> Path:
    """Build a path inside `data/raw/`."""

    return RAW_DATA_DIR / filename


def interim_data_path(filename: str) -> Path:
    """Build a path inside `data/interim/`."""

    return INTERIM_DATA_DIR / filename


def processed_data_path(filename: str) -> Path:
    """Build a path inside `data/processed/`."""

    return PROCESSED_DATA_DIR / filename


def figure_path(filename: str) -> Path:
    """Build a path inside `outputs/figures/`."""

    return FIGURES_DIR / filename


def table_path(filename: str) -> Path:
    """Build a path inside `outputs/tables/`."""

    return TABLES_DIR / filename


def doc_path(filename: str) -> Path:
    """Build a path inside `docs/`."""

    return DOCS_DIR / filename
