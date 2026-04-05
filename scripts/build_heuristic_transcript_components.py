"""Build a heuristic component-level transcript export from `FINAL.csv`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.buyback_events import build_buyback_pattern
from src.data.heuristic_components import build_component_dataset_from_transcripts
from src.data.load_transcripts import resolve_transcript_path


DEFAULT_USECOLS = [
    "transcriptid",
    "companyid",
    "companyname",
    "ticker",
    "call_date",
    "full_transcript_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build heuristic component-level transcript rows from FINAL.csv."
    )
    parser.add_argument(
        "--all-transcripts",
        action="store_true",
        help="Process all transcripts instead of only buyback-related transcripts.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="CSV chunk size for streaming the large FINAL.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = PROJECT_ROOT / "data" / "interim" / "transcript_components.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    transcript_path = resolve_transcript_path()
    buyback_pattern = build_buyback_pattern()
    component_count = 0
    chunk_count = 0
    matched_transcript_count = 0
    wrote_header = False
    counts_by_type: dict[tuple[int, int], int] = {}

    for chunk in pd.read_csv(transcript_path, usecols=DEFAULT_USECOLS, chunksize=args.chunksize):
        chunk_count += 1
        if not args.all_transcripts:
            text_series = chunk["full_transcript_text"].fillna("").astype(str)
            chunk = chunk.loc[text_series.str.contains(buyback_pattern, na=False)].copy()
        if chunk.empty:
            continue

        matched_transcript_count += len(chunk)
        component_rows = build_component_dataset_from_transcripts(chunk)
        component_count += len(component_rows)

        grouped_counts = (
            component_rows[["transcriptcomponenttypeid", "speakertypeid"]]
            .value_counts(dropna=False)
            .to_dict()
        )
        for key, value in grouped_counts.items():
            counts_by_type[key] = counts_by_type.get(key, 0) + int(value)

        component_rows.to_csv(
            output_path,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

        if chunk_count % 25 == 0:
            print(
                f"Processed {chunk_count} chunks | matched transcripts: {matched_transcript_count:,} "
                f"| component rows written: {component_count:,}"
            )

    if not wrote_header:
        pd.DataFrame(
            columns=[
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
        ).to_csv(output_path, index=False)

    print(f"Wrote {component_count:,} heuristic component rows to {output_path}")
    print(f"Matched transcripts: {matched_transcript_count:,}")
    print("Counts by (transcriptcomponenttypeid, speakertypeid):")
    for key in sorted(counts_by_type):
        print(f"  {key}: {counts_by_type[key]:,}")


if __name__ == "__main__":
    main()
