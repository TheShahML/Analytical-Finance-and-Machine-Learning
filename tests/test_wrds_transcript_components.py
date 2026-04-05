"""Tests for true WRDS-backed transcript component extraction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_transcript_components import (
    load_transcript_components,
    resolve_transcript_component_path,
)
from src.data.wrds_transcript_components import (
    ComponentQueryPlan,
    build_component_query,
    build_component_query_plan,
    export_wrds_transcript_components,
)


class FakeWrdsConnection:
    """Minimal fake WRDS connection that records SQL batches."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def raw_sql(self, query: str) -> pd.DataFrame:
        self.queries.append(query)

        if "information_schema.columns" in query:
            return pd.DataFrame(
                {
                    "table_schema": ["ciq"] * 5,
                    "table_name": ["wrds_transcript_component"] * 5,
                    "column_name": [
                        "transcriptid",
                        "componentorder",
                        "componenttext",
                        "transcriptcomponenttypeid",
                        "speakertypeid",
                    ],
                }
            )

        if "IN (101,102)" in query:
            return pd.DataFrame(
                {
                    "transcriptid": [101, 101, 102],
                    "componentorder": [1, 2, 1],
                    "componenttext": ["Prompt", "Question", "Answer"],
                    "transcriptcomponenttypeid": [3, 3, 3],
                    "speakertypeid": [0, 3, 1],
                }
            )

        if "IN (103)" in query:
            return pd.DataFrame(
                {
                    "transcriptid": [103],
                    "componentorder": [1],
                    "componenttext": ["Prepared remarks"],
                    "transcriptcomponenttypeid": [2],
                    "speakertypeid": [1],
                }
            )

        return pd.DataFrame(
            columns=[
                "transcriptid",
                "componentorder",
                "componenttext",
                "transcriptcomponenttypeid",
                "speakertypeid",
            ]
        )


def test_build_component_query_requests_true_component_fields() -> None:
    plan = ComponentQueryPlan(
        description="direct test plan",
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
    query = build_component_query([101, 102], plan)

    assert "transcriptcomponenttypeid" in query
    assert "speakertypeid" in query
    assert "ciq.wrds_transcript_component" in query
    assert "IN (101,102)" in query


def test_build_component_query_plan_prefers_direct_wrds_table() -> None:
    schema = pd.DataFrame(
        {
            "table_schema": ["ciq"] * 8,
            "table_name": [
                "ciqtranscriptcomponent",
                "ciqtranscriptcomponent",
                "ciqtranscriptcomponent",
                "wrds_transcript_component",
                "wrds_transcript_component",
                "wrds_transcript_component",
                "wrds_transcript_component",
                "wrds_transcript_component",
            ],
            "column_name": [
                "transcriptid",
                "componentorder",
                "componenttext",
                "transcriptid",
                "componentorder",
                "componenttext",
                "transcriptcomponenttypeid",
                "speakertypeid",
            ],
        }
    )

    plan = build_component_query_plan(schema)

    assert plan.base_table == "ciq.wrds_transcript_component"
    assert "direct component pull" in plan.description


def test_export_wrds_transcript_components_batches_and_enriches(tmp_path: Path, monkeypatch) -> None:
    metadata = pd.DataFrame(
        {
            "transcriptid": [101, 102, 103],
            "companyid": [1, 1, 2],
            "companyname": ["Example A", "Example A", "Example B"],
            "ticker": ["AAA", "AAA", "BBB"],
            "call_date": ["2024-01-31", "2024-04-30", "2024-07-31"],
        }
    )
    fake_db = FakeWrdsConnection()
    output_path = tmp_path / "wrds_transcript_components.csv"

    monkeypatch.setattr(
        "src.data.wrds_transcript_components.load_transcript_metadata",
        lambda transcript_path=None, nrows=None: metadata,
    )

    summary = export_wrds_transcript_components(
        fake_db,
        output_path=output_path,
        batch_size=2,
        progress_every=0,
    )

    exported = pd.read_csv(output_path)

    assert summary.transcript_count == 3
    assert summary.component_row_count == 4
    assert summary.batch_count == 2
    assert list(exported["component_source"].unique()) == ["wrds"]
    assert list(exported.columns) == [
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
    ]
    assert len(fake_db.queries) == 3


def test_resolve_transcript_component_path_prefers_env_override(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "true_components.csv"
    env_file.write_text("transcriptid,componenttext\n1,hello\n")
    monkeypatch.setenv("TRANSCRIPT_COMPONENTS_PATH", str(env_file))

    resolved = resolve_transcript_component_path()

    assert resolved == env_file


def test_load_transcript_components_filters_to_requested_transcript_ids(tmp_path: Path) -> None:
    component_file = tmp_path / "components.csv"
    pd.DataFrame(
        {
            "transcriptid": [101, 101, 202],
            "componentorder": [1, 2, 1],
            "componenttext": ["Intro", "Question", "Other call"],
            "transcriptcomponenttypeid": [2, 3, 2],
            "speakertypeid": [2, 3, 2],
        }
    ).to_csv(component_file, index=False)

    loaded = load_transcript_components(component_file, transcript_ids=[101])

    assert loaded["transcriptid"].tolist() == [101, 101]
    assert loaded["componenttext"].tolist() == ["Intro", "Question"]
