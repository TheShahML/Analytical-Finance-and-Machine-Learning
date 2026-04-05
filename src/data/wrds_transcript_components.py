"""WRDS-backed puller for true Capital IQ transcript component rows.

This module turns the one-off provenance logic from notebook 00 into a reusable
extractor for component-level transcript rows keyed by `transcriptid`. It is
meant to produce a high-trust alternative to the heuristic component parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Iterable, Sequence

import pandas as pd

from src.config.settings import DATA_DIR
from src.data.load_transcripts import get_available_columns, load_raw_transcripts, resolve_transcript_path


DEFAULT_COMPONENT_OUTPUT_PATH = DATA_DIR / "interim" / "wrds_transcript_components.csv"
DEFAULT_TRANSCRIPT_METADATA_COLUMNS = (
    "transcriptid",
    "companyid",
    "companyname",
    "ticker",
    "call_date",
)
DEFAULT_COMPONENT_SELECT_COLUMNS = (
    "transcriptid",
    "componentorder",
    "componenttext",
    "transcriptcomponenttypeid",
    "speakertypeid",
)
DEFAULT_COMPONENT_OUTPUT_COLUMNS = (
    "transcriptid",
    "componentorder",
    "componenttext",
    "transcriptcomponenttypeid",
    "speakertypeid",
    "speakername",
    "component_source",
    "companyid",
    "companyname",
    "ticker",
    "call_date",
)
CIQ_SCHEMA = "ciq"
BASE_COMPONENT_TABLE = f"{CIQ_SCHEMA}.ciqtranscriptcomponent"
TARGET_COMPONENT_COLUMNS = (
    "transcriptid",
    "componentorder",
    "componenttext",
    "transcriptcomponenttypeid",
    "speakertypeid",
    "speakername",
)


@dataclass(frozen=True)
class WrdsTranscriptComponentSummary:
    """Compact summary returned by a WRDS component export run."""

    transcript_count: int
    component_row_count: int
    output_path: Path
    batch_count: int


@dataclass(frozen=True)
class ComponentQueryPlan:
    """Resolved WRDS query plan for transcript-component extraction."""

    description: str
    base_table: str
    joins: tuple[str, ...]
    select_expressions: tuple[str, ...]


class ComponentSchemaError(RuntimeError):
    """Raised when the WRDS CIQ schema cannot support a component pull."""


def default_component_query_plan() -> ComponentQueryPlan:
    """Return the preferred direct WRDS transcript-component query plan."""

    return ComponentQueryPlan(
        description="direct component pull from ciq.wrds_transcript_component",
        base_table="ciq.wrds_transcript_component",
        joins=(),
        select_expressions=(
            "transcriptid",
            "componentorder",
            "componenttext",
            "transcriptcomponenttypeid",
            "speakertypeid",
            "NULL::text AS speakername",
        ),
    )


def _chunked(values: Sequence[int], chunk_size: int) -> Iterable[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])


def _import_wrds():
    try:
        import wrds  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in integration use
        raise ImportError(
            "The `wrds` package is required for true transcript component pulls. "
            "Install it with `pip install wrds` or `pip install -r requirements.txt`."
        ) from exc
    return wrds


def connect_wrds(wrds_username: str | None = None):
    """Create a WRDS connection using an explicit username or env fallback."""

    wrds = _import_wrds()
    username = wrds_username or os.getenv("WRDS_USERNAME")
    return wrds.Connection(wrds_username=username)


def load_transcript_metadata(
    transcript_path: str | Path | None = None,
    *,
    columns: Sequence[str] = DEFAULT_TRANSCRIPT_METADATA_COLUMNS,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load canonical transcript metadata needed to enrich component rows."""

    resolved_path = resolve_transcript_path(transcript_path)
    available = set(get_available_columns(resolved_path))
    selected_columns = [column for column in columns if column in available]

    transcripts = load_raw_transcripts(path=resolved_path, columns=selected_columns or None, nrows=nrows)
    if "transcriptid" not in transcripts.columns:
        raise KeyError("Canonical transcript data must include `transcriptid`.")

    metadata = transcripts.loc[:, [column for column in columns if column in transcripts.columns]].copy()
    metadata = metadata.dropna(subset=["transcriptid"]).drop_duplicates(subset=["transcriptid"])
    return metadata.reset_index(drop=True)


def get_ciq_transcript_schema(db) -> pd.DataFrame:
    """Return CIQ transcript-related columns available in WRDS."""

    query = """
        SELECT
            table_schema,
            table_name,
            column_name
        FROM information_schema.columns
        WHERE table_schema = 'ciq'
          AND table_name ILIKE '%%transcript%%'
        ORDER BY table_name, ordinal_position
    """
    schema = db.raw_sql(query)
    if schema.empty:
        raise ComponentSchemaError("No CIQ transcript-related tables were found in WRDS.")
    return schema


def summarize_ciq_transcript_schema(schema_df: pd.DataFrame) -> str:
    """Return a compact text summary of candidate CIQ transcript tables."""

    lines: list[str] = []
    grouped = schema_df.groupby(["table_schema", "table_name"])["column_name"].apply(list)
    for (schema_name, table_name), columns in grouped.items():
        relevant = [column for column in columns if "transcript" in column or "speaker" in column or "component" in column]
        preview = ", ".join(relevant[:12] if relevant else columns[:12])
        lines.append(f"{schema_name}.{table_name}: {preview}")
    return "\n".join(lines)


def _table_column_map(schema_df: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for row in schema_df.itertuples(index=False):
        table_ref = f"{row.table_schema}.{row.table_name}"
        mapping.setdefault(table_ref, set()).add(row.column_name)
    return mapping


def _table_priority(table_ref: str) -> tuple[int, int, str]:
    lower = table_ref.lower()
    return (
        0 if ".wrds_" in lower else 1,
        0 if "component" in lower else 1,
        lower,
    )


def _choose_best_table(table_candidates: dict[str, set[str]], required: set[str]) -> str | None:
    valid = [table_ref for table_ref, columns in table_candidates.items() if required.issubset(columns)]
    if not valid:
        return None
    return sorted(valid, key=_table_priority)[0]


def build_component_query_plan(schema_df: pd.DataFrame) -> ComponentQueryPlan:
    """Infer the best available WRDS query plan for transcript components."""

    table_columns = _table_column_map(schema_df)

    direct_required = {
        "transcriptid",
        "componentorder",
        "componenttext",
        "transcriptcomponenttypeid",
        "speakertypeid",
    }
    direct_table = _choose_best_table(table_columns, direct_required)
    if direct_table is not None:
        has_speakername = "speakername" in table_columns[direct_table]
        speaker_expr = "speakername" if has_speakername else "NULL::text AS speakername"
        return ComponentQueryPlan(
            description=f"direct component pull from {direct_table}",
            base_table=direct_table,
            joins=(),
            select_expressions=(
                "transcriptid",
                "componentorder",
                "componenttext",
                "transcriptcomponenttypeid",
                "speakertypeid",
                speaker_expr,
            ),
        )

    base_columns = table_columns.get(BASE_COMPONENT_TABLE)
    if base_columns is None:
        raise ComponentSchemaError(
            "The base table `ciq.ciqtranscriptcomponent` is not available.\n"
            f"{summarize_ciq_transcript_schema(schema_df)}"
        )

    if not {"transcriptid", "componentorder", "componenttext"}.issubset(base_columns):
        raise ComponentSchemaError(
            "`ciq.ciqtranscriptcomponent` is missing one of `transcriptid`, `componentorder`, or `componenttext`.\n"
            f"{summarize_ciq_transcript_schema(schema_df)}"
        )

    joins: list[str] = []
    select_expressions = [
        "base.transcriptid",
        "base.componentorder",
        "base.componenttext",
    ]

    transcriptcomponentid_join = "transcriptcomponentid" in base_columns
    join_key_candidates = []
    if transcriptcomponentid_join:
        join_key_candidates.append(("transcriptcomponentid", "base.transcriptcomponentid = {alias}.transcriptcomponentid"))
    join_key_candidates.append(
        ("transcriptid_componentorder", "base.transcriptid = {alias}.transcriptid AND base.componentorder = {alias}.componentorder")
    )

    def find_join_table(target_column: str) -> tuple[str, str] | None:
        for join_key_name, join_template in join_key_candidates:
            if join_key_name == "transcriptcomponentid":
                required = {"transcriptcomponentid", target_column}
            else:
                required = {"transcriptid", "componentorder", target_column}
            table_ref = _choose_best_table(table_columns, required)
            if table_ref is not None:
                return table_ref, join_template
        return None

    type_target = find_join_table("transcriptcomponenttypeid")
    if type_target is None:
        raise ComponentSchemaError(
            "Could not find a CIQ transcript table containing `transcriptcomponenttypeid`.\n"
            f"{summarize_ciq_transcript_schema(schema_df)}"
        )
    type_table, type_join_template = type_target
    joins.append(f"LEFT JOIN {type_table} type_map ON {type_join_template.format(alias='type_map')}")
    select_expressions.append("type_map.transcriptcomponenttypeid")

    speaker_target = find_join_table("speakertypeid")
    if speaker_target is None:
        raise ComponentSchemaError(
            "Could not find a CIQ transcript table containing `speakertypeid`.\n"
            f"{summarize_ciq_transcript_schema(schema_df)}"
        )
    speaker_table, speaker_join_template = speaker_target
    joins.append(f"LEFT JOIN {speaker_table} speaker_map ON {speaker_join_template.format(alias='speaker_map')}")
    select_expressions.append("speaker_map.speakertypeid")

    speakername_target = find_join_table("speakername")
    if speakername_target is not None:
        speakername_table, speakername_join_template = speakername_target
        joins.append(
            f"LEFT JOIN {speakername_table} speaker_name_map ON {speakername_join_template.format(alias='speaker_name_map')}"
        )
        select_expressions.append("speaker_name_map.speakername")
    else:
        select_expressions.append("NULL::text AS speakername")

    return ComponentQueryPlan(
        description=(
            f"joined component pull from {BASE_COMPONENT_TABLE} + {type_table} + {speaker_table}"
        ),
        base_table=BASE_COMPONENT_TABLE,
        joins=tuple(joins),
        select_expressions=tuple(select_expressions),
    )


def build_component_query(transcript_ids: Sequence[int], plan: ComponentQueryPlan) -> str:
    """Return the WRDS SQL query for a transcript-component batch."""

    if not transcript_ids:
        raise ValueError("transcript_ids must not be empty.")

    id_list = ",".join(str(int(transcript_id)) for transcript_id in transcript_ids)
    return f"""
        SELECT
            {", ".join(plan.select_expressions)}
        FROM {plan.base_table} base
        {' '.join(plan.joins)}
        WHERE base.transcriptid IN ({id_list})
          AND base.componenttext IS NOT NULL
        ORDER BY base.transcriptid, base.componentorder
    """


def fetch_component_batch(db, transcript_ids: Sequence[int], plan: ComponentQueryPlan) -> pd.DataFrame:
    """Fetch one batch of transcript-component rows from WRDS."""

    if not transcript_ids:
        return pd.DataFrame(columns=DEFAULT_COMPONENT_SELECT_COLUMNS)

    batch = db.raw_sql(build_component_query(transcript_ids, plan))
    if batch.empty:
        return pd.DataFrame(columns=DEFAULT_COMPONENT_OUTPUT_COLUMNS)

    expected = [column for column in DEFAULT_COMPONENT_OUTPUT_COLUMNS if column in batch.columns]
    return batch.loc[:, expected].copy()


def enrich_component_rows(component_rows: pd.DataFrame, transcript_metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical metadata and stable source markers to component rows."""

    if component_rows.empty:
        return pd.DataFrame(columns=list(DEFAULT_COMPONENT_OUTPUT_COLUMNS))

    enriched = component_rows.merge(
        transcript_metadata,
        on="transcriptid",
        how="left",
    )
    if "speakername" not in enriched.columns:
        enriched["speakername"] = pd.NA
    enriched["component_source"] = "wrds"
    ordered_columns = [column for column in DEFAULT_COMPONENT_OUTPUT_COLUMNS if column in enriched.columns]
    return enriched.loc[:, ordered_columns].copy()


def export_wrds_transcript_components(
    db,
    *,
    transcript_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_COMPONENT_OUTPUT_PATH,
    batch_size: int = 200,
    nrows: int | None = None,
    pause_seconds: float = 0.0,
    progress_every: int = 25,
) -> WrdsTranscriptComponentSummary:
    """Pull and export true transcript-component rows for the canonical sample."""

    transcript_metadata = load_transcript_metadata(transcript_path, nrows=nrows)
    transcript_ids = transcript_metadata["transcriptid"].dropna().astype(int).tolist()
    try:
        schema_df = get_ciq_transcript_schema(db)
        plan = build_component_query_plan(schema_df)
    except Exception as exc:
        print(
            "Schema discovery failed; falling back to the default WRDS component table plan. "
            f"Underlying error: {exc}"
        )
        plan = default_component_query_plan()
    print(f"Using WRDS transcript component query plan: {plan.description}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    total_rows = 0
    batch_count = 0
    wrote_header = False

    for batch_ids in _chunked(transcript_ids, batch_size):
        batch_count += 1
        batch = fetch_component_batch(db, batch_ids, plan)
        enriched = enrich_component_rows(batch, transcript_metadata)

        if not enriched.empty:
            total_rows += len(enriched)
            enriched.to_csv(destination, mode="a", header=not wrote_header, index=False)
            wrote_header = True

        if progress_every and batch_count % progress_every == 0:
            print(
                f"Processed {batch_count} batches | transcripts: {min(batch_count * batch_size, len(transcript_ids)):,}/"
                f"{len(transcript_ids):,} | component rows written: {total_rows:,}"
            )

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not wrote_header:
        pd.DataFrame(columns=list(DEFAULT_COMPONENT_OUTPUT_COLUMNS)).to_csv(destination, index=False)

    return WrdsTranscriptComponentSummary(
        transcript_count=len(transcript_ids),
        component_row_count=total_rows,
        output_path=destination,
        batch_count=batch_count,
    )
