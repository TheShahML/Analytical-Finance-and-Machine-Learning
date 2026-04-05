#!/usr/bin/env python3
"""Compare keyword-based and event-based buyback transcript samples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import CANONICAL_TRANSCRIPT_PATH, PROJECT_ROOT
from src.data.buyback_events import match_buyback_events_to_transcripts


DEFAULT_KEYWORD_SETS: dict[str, list[str]] = {
    "current_default": [
        "buyback",
        "buy back",
        "repurchase",
        "share repurchase",
        "repurchasing",
        "stock repurchase",
        "repurchase program",
        "repurchase plan",
    ],
    "strict_equity": [
        "buyback",
        "buy back",
        "share repurchase",
        "stock repurchase",
        "repurchase program",
        "repurchase plan",
    ],
    "generic_repurchase": [
        "repurchase",
        "repurchasing",
    ],
}

DEFAULT_EVENT_TYPE_IDS = (36, 152, 230, 231, 232, 233, 234)
DEFAULT_INPUT_CANDIDATES = (
    PROJECT_ROOT / "data" / "processed" / "01_data_audit_cleaned_transcripts.parquet",
    PROJECT_ROOT / "data" / "processed" / "01_data_audit_usable_transcripts.parquet",
    CANONICAL_TRANSCRIPT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrds-username", type=str, default=None, help="WRDS username for event-based sample pull.")
    parser.add_argument("--input-path", type=Path, default=None, help="Optional transcript panel override.")
    parser.add_argument("--window-days", type=int, default=2, help="Match window between event date and transcript date.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="CSV scan chunk size.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/reports/buyback_sample_comparison.json"),
        help="Optional output JSON path.",
    )
    return parser


def normalize_phrases(phrases: Iterable[str]) -> list[str]:
    return [phrase.lower().strip() for phrase in phrases if phrase.strip()]


def resolve_input_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Transcript panel not found: {resolved}")
        return resolved

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No transcript panel source was found for buyback sample comparison.")


def load_transcript_metadata(path: str | Path) -> pd.DataFrame:
    resolved_path = resolve_input_path(path)
    if resolved_path.suffix.lower() == ".parquet":
        return pd.read_parquet(
            resolved_path,
            columns=["transcriptid", "companyid", "companyname", "call_date"],
        )
    return pd.read_csv(
        resolved_path,
        usecols=["transcriptid", "companyid", "companyname", "call_date"],
        parse_dates=["call_date"],
    )


def stream_keyword_sets(
    path: str | Path,
    *,
    chunk_size: int = 2000,
) -> tuple[dict[str, set[int]], dict[str, set[int]], int]:
    normalized_sets = {
        name: normalize_phrases(keywords)
        for name, keywords in DEFAULT_KEYWORD_SETS.items()
    }
    individual_keywords = DEFAULT_KEYWORD_SETS["current_default"]
    normalized_individual = {
        keyword: normalize_phrases([keyword])[0]
        for keyword in individual_keywords
    }

    sample_sets = {name: set() for name in DEFAULT_KEYWORD_SETS}
    keyword_hits = {keyword: set() for keyword in individual_keywords}
    rows_scanned = 0

    resolved_path = resolve_input_path(path)
    if resolved_path.suffix.lower() == ".parquet":
        chunk = pd.read_parquet(
            resolved_path,
            columns=["transcriptid", "full_transcript_text"],
        )
        chunks = [chunk]
    else:
        chunks = pd.read_csv(
            resolved_path,
            usecols=["transcriptid", "full_transcript_text"],
            chunksize=chunk_size,
        )

    for chunk in chunks:
        rows_scanned += len(chunk)
        ids = pd.to_numeric(chunk["transcriptid"], errors="coerce")
        text = chunk["full_transcript_text"].fillna("").astype(str).str.lower()

        for name, phrases in normalized_sets.items():
            mask = pd.Series(False, index=text.index)
            for phrase in phrases:
                mask = mask | text.str.contains(phrase, regex=False, na=False)
            sample_sets[name].update(ids.loc[mask].dropna().astype(int).tolist())

        for keyword, phrase in normalized_individual.items():
            mask = text.str.contains(phrase, regex=False, na=False)
            keyword_hits[keyword].update(ids.loc[mask].dropna().astype(int).tolist())

    return sample_sets, keyword_hits, rows_scanned


def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def pull_buyback_events_from_wrds(
    *,
    wrds_username: str,
    company_ids: list[int],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    event_type_ids: tuple[int, ...] = DEFAULT_EVENT_TYPE_IDS,
    company_chunk_size: int = 500,
) -> pd.DataFrame:
    import wrds

    db = wrds.Connection(wrds_username=wrds_username, autoconnect=True, verbose=False)
    batches: list[pd.DataFrame] = []
    event_ids_sql = ",".join(str(event_id) for event_id in event_type_ids)
    min_date_str = min_date.strftime("%Y-%m-%d")
    max_date_str = max_date.strftime("%Y-%m-%d")

    try:
        for batch_company_ids in _chunked(company_ids, company_chunk_size):
            company_ids_sql = ",".join(str(company_id) for company_id in batch_company_ids)
            query = f"""
                SELECT
                    keydevid,
                    companyid,
                    companyname,
                    headline,
                    eventtype AS event_type,
                    keydeveventtypeid,
                    announcedate AS event_date,
                    announceddateutc,
                    mostimportantdateutc
                FROM ciq.wrds_keydev
                WHERE companyid IN ({company_ids_sql})
                  AND keydeveventtypeid IN ({event_ids_sql})
                  AND announcedate BETWEEN '{min_date_str}' AND '{max_date_str}'
                ORDER BY companyid, announcedate, keydevid
            """
            batch = db.raw_sql(query, date_cols=["event_date", "announceddateutc", "mostimportantdateutc"])
            if not batch.empty:
                batches.append(batch)
    finally:
        db.close()

    if not batches:
        return pd.DataFrame(
            columns=[
                "keydevid",
                "companyid",
                "companyname",
                "headline",
                "event_type",
                "keydeveventtypeid",
                "event_date",
                "announceddateutc",
                "mostimportantdateutc",
            ]
        )

    events = pd.concat(batches, ignore_index=True)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    return events.drop_duplicates(subset=["keydevid"]).reset_index(drop=True)


def summarize_overlap(left: set[int], right: set[int]) -> dict[str, Any]:
    union = left | right
    overlap = left & right
    left_only = left - right
    right_only = right - left

    return {
        "left_count": len(left),
        "right_count": len(right),
        "overlap_count": len(overlap),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "jaccard_similarity_pct": round(100 * len(overlap) / len(union), 2) if union else 0.0,
        "different_pct_of_union": round(100 * len(left_only | right_only) / len(union), 2) if union else 0.0,
        "overlap_as_pct_of_left": round(100 * len(overlap) / len(left), 2) if left else 0.0,
        "overlap_as_pct_of_right": round(100 * len(overlap) / len(right), 2) if right else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolve_input_path(args.input_path)
    print(f"[compare] Using transcript panel: {input_path}", flush=True)

    transcript_metadata = load_transcript_metadata(input_path)
    transcript_metadata["transcriptid"] = pd.to_numeric(transcript_metadata["transcriptid"], errors="coerce").astype("Int64")
    transcript_metadata["companyid"] = pd.to_numeric(transcript_metadata["companyid"], errors="coerce").astype("Int64")
    transcript_metadata["call_date"] = pd.to_datetime(transcript_metadata["call_date"], errors="coerce").dt.normalize()

    print("[compare] Building keyword-based samples...", flush=True)
    sample_sets, keyword_hits, rows_scanned = stream_keyword_sets(input_path, chunk_size=args.chunk_size)
    print(
        f"[compare] Keyword scan complete. Rows scanned: {rows_scanned:,}. "
        f"Current default hits: {len(sample_sets['current_default']):,}.",
        flush=True,
    )

    event_transcript_ids: set[int] = set()
    event_summary: dict[str, Any] = {"status": "not_run"}
    if args.wrds_username:
        print(f"[compare] Pulling structured buyback events from WRDS as {args.wrds_username}...", flush=True)
        company_ids = (
            transcript_metadata["companyid"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        min_date = transcript_metadata["call_date"].dropna().min() - pd.Timedelta(days=args.window_days)
        max_date = transcript_metadata["call_date"].dropna().max() + pd.Timedelta(days=args.window_days)
        events = pull_buyback_events_from_wrds(
            wrds_username=args.wrds_username,
            company_ids=company_ids,
            min_date=min_date,
            max_date=max_date,
        )
        matched = match_buyback_events_to_transcripts(
            events,
            transcript_metadata.dropna(subset=["companyid", "call_date"]).assign(
                companyid=lambda df: df["companyid"].astype(int),
                transcriptid=lambda df: df["transcriptid"].astype(int),
            ),
            window_days=args.window_days,
        )
        event_transcript_ids = set(pd.to_numeric(matched["transcriptid"], errors="coerce").dropna().astype(int))
        event_summary = {
            "status": "ok",
            "event_rows": int(len(events)),
            "distinct_keydevid": int(events["keydevid"].nunique()) if not events.empty else 0,
            "matched_transcripts": int(len(event_transcript_ids)),
            "event_type_counts": (
                events["event_type"].astype(str).value_counts().to_dict()
                if not events.empty else {}
            ),
        }
        print(
            f"[compare] Event pull complete. Event rows: {event_summary['event_rows']:,} | "
            f"matched transcripts: {event_summary['matched_transcripts']:,}.",
            flush=True,
        )

    comparisons = {
        "strict_vs_current": summarize_overlap(
            sample_sets["current_default"],
            sample_sets["strict_equity"],
        ),
        "generic_vs_current": summarize_overlap(
            sample_sets["current_default"],
            sample_sets["generic_repurchase"],
        ),
    }
    if event_transcript_ids:
        comparisons["event_vs_current"] = summarize_overlap(
            event_transcript_ids,
            sample_sets["current_default"],
        )
        comparisons["event_vs_strict"] = summarize_overlap(
            event_transcript_ids,
            sample_sets["strict_equity"],
        )
        comparisons["event_vs_generic"] = summarize_overlap(
            event_transcript_ids,
            sample_sets["generic_repurchase"],
        )
        comparisons["intersection_all_three_count"] = len(
            event_transcript_ids
            & sample_sets["current_default"]
            & sample_sets["strict_equity"]
        )

    summary = {
        "rows_scanned": rows_scanned,
        "input_path": str(input_path),
        "keyword_sample_counts": {name: len(values) for name, values in sample_sets.items()},
        "individual_keyword_counts": {name: len(values) for name, values in keyword_hits.items()},
        "event_sample": event_summary,
        "comparisons": comparisons,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
