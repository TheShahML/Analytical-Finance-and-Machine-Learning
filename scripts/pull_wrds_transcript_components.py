"""Pull true component-level Capital IQ transcript rows from WRDS.

The canonical transcript-level file stays `FINAL.csv`. This script uses the
`transcriptid` values already present there to fetch true component rows with
`transcriptcomponenttypeid` and `speakertypeid`, then writes a preferred
component export for downstream Q&A splitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wrds_transcript_components import (
    DEFAULT_COMPONENT_OUTPUT_PATH,
    connect_wrds,
    export_wrds_transcript_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull true transcript component rows from WRDS for the transcript IDs in FINAL.csv."
    )
    parser.add_argument(
        "--wrds-username",
        help="Optional WRDS username override. Falls back to WRDS_USERNAME in the environment.",
    )
    parser.add_argument(
        "--transcript-path",
        help="Optional override for the canonical transcript-level input file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_COMPONENT_OUTPUT_PATH),
        help="Where to write the true component export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Transcript IDs per WRDS batch query.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        help="Optional limit for transcript metadata rows during testing.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between WRDS batch queries.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Connecting to WRDS for true transcript component pull...")
    db = connect_wrds(args.wrds_username)

    try:
        summary = export_wrds_transcript_components(
            db,
            transcript_path=args.transcript_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            nrows=args.nrows,
            pause_seconds=args.pause_seconds,
            progress_every=args.progress_every,
        )
    finally:
        db.close()

    print(
        f"Wrote {summary.component_row_count:,} true WRDS component rows for "
        f"{summary.transcript_count:,} transcripts to {summary.output_path}"
    )
    print(f"Batches executed: {summary.batch_count:,}")


if __name__ == "__main__":
    main()
