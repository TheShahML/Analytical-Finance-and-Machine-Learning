#!/usr/bin/env python3
"""Run a lightweight buyback sentiment/clarity smoke test outside Jupyter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - runtime convenience
    torch = None

from src.analysis.binning import create_sentiment_clarity_matrix
from src.config.settings import ProjectPaths
from src.data.buyback_events import build_buyback_pattern, extract_buyback_sentences, identify_buyback_transcripts
from src.data.load_transcript_components import (
    component_data_supports_qa_split,
    load_transcript_components,
    resolve_transcript_component_path,
)
from src.data.load_transcripts import load_raw_transcripts
from src.data.qa_split import (
    flag_suspicious_qa_pairs,
    pair_questions_responses,
    split_prepared_qa,
    summarize_qa_pair_quality,
)
from src.data.revenue_surprise import bucket_revenue, compute_trend_revenue_surprise, merge_revenue_surprise
from src.features.clarity import (
    bucket_clarity,
    compute_clarity_composite,
    compute_hedge_density,
    compute_modified_fog,
    compute_qa_relevance,
    compute_specificity,
)
from src.features.finbert_sentiment import bucket_sentiment, load_finbert_pipeline, score_transcript_sections
from src.finance.event_study import (
    IMAGE_SPEC_ESTIMATION_WINDOW,
    IMAGE_SPEC_ROBUSTNESS_WINDOWS,
    compute_caar_by_bins,
    run_event_study_from_wide_returns,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-rows", type=int, default=3000, help="Rows to read from FINAL.csv.")
    parser.add_argument("--buyback-calls", type=int, default=40, help="Buyback calls to keep in the smoke test.")
    parser.add_argument(
        "--max-clarity-pairs",
        type=int,
        default=75,
        help="Maximum clean buyback Q&A pairs to score for clarity relevance.",
    )
    parser.add_argument("--finbert-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Override model device, e.g. cpu or cuda.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/reports/buyback_sample_summary.json"),
        help="Where to write the smoke-test summary JSON.",
    )
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=Path("outputs/tables/buyback_sample_matrix.csv"),
        help="Where to write the sentiment x clarity sample matrix CSV.",
    )
    return parser


def resolve_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    args = build_parser().parse_args()
    paths = ProjectPaths()
    device = resolve_device(args.device)

    print(f"[sample] Loading {args.transcript_rows:,} transcript rows from FINAL.csv...", flush=True)
    transcripts = load_raw_transcripts(nrows=args.transcript_rows)
    print(f"[sample] Loaded {len(transcripts):,} transcript rows.", flush=True)
    buyback_sample = identify_buyback_transcripts(transcripts).copy()
    buyback_sample = buyback_sample.head(args.buyback_calls).copy()
    if buyback_sample.empty:
        raise SystemExit("No buyback transcripts were found in the requested sample.")

    buyback_sample["buyback_sentences"] = buyback_sample["full_transcript_text"].map(extract_buyback_sentences)
    buyback_sample["calendar_quarter"] = buyback_sample["call_date"].dt.to_period("Q").astype(str)
    print(f"[sample] Working with {len(buyback_sample):,} buyback calls.", flush=True)

    component_path = None
    component_rows = pd.DataFrame()
    try:
        component_path = resolve_transcript_component_path()
        print(f"[sample] Loading component rows from {component_path}...", flush=True)
        component_rows = load_transcript_components(
            component_path,
            transcript_ids=buyback_sample["transcriptid"].dropna().unique().tolist(),
        )
    except FileNotFoundError:
        component_rows = pd.DataFrame()
    print(f"[sample] Loaded {len(component_rows):,} component rows.", flush=True)

    prepared_df = pd.DataFrame()
    qa_df = pd.DataFrame()
    qa_pairs = pd.DataFrame(columns=["transcriptid", "question_text", "response_text"])
    qa_pair_quality = {
        "pair_count": 0,
        "suspicious_pairs": 0,
        "suspicious_share": 0.0,
        "median_question_words": 0.0,
        "median_response_words": 0.0,
    }
    if not component_rows.empty and component_data_supports_qa_split(component_rows):
        prepared_df, qa_df = split_prepared_qa(component_rows)
        qa_pairs = pair_questions_responses(qa_df)
        qa_pair_quality = summarize_qa_pair_quality(qa_pairs)
        print(
            f"[sample] Prepared rows: {len(prepared_df):,} | Q&A rows: {len(qa_df):,} | pairs: {len(qa_pairs):,}.",
            flush=True,
        )

    prepared_text_by_transcript = (
        prepared_df.groupby("transcriptid")["componenttext"].apply(lambda values: " ".join(values.dropna().astype(str)))
        if not prepared_df.empty and "componenttext" in prepared_df.columns
        else pd.Series(dtype=object)
    )
    qa_text_by_transcript = (
        qa_df.groupby("transcriptid")["componenttext"].apply(lambda values: " ".join(values.dropna().astype(str)))
        if not qa_df.empty and "componenttext" in qa_df.columns
        else pd.Series(dtype=object)
    )

    print(f"[sample] Loading FinBERT on {device}...", flush=True)
    finbert_pipeline = load_finbert_pipeline(device=device, batch_size=args.finbert_batch_size)
    print("[sample] Scoring transcript sentiment...", flush=True)
    sentiment_rows: list[dict[str, float | int]] = []
    for row in buyback_sample.itertuples(index=False):
        transcript_id = getattr(row, "transcriptid")
        sentiment_rows.append(
            {
                "transcriptid": transcript_id,
                **score_transcript_sections(
                    transcript=getattr(row, "full_transcript_text", ""),
                    prep_text=prepared_text_by_transcript.get(transcript_id, ""),
                    qa_text=qa_text_by_transcript.get(transcript_id, ""),
                    buyback_sentences=getattr(row, "buyback_sentences", []),
                    pipeline_obj=finbert_pipeline,
                    device=device,
                    batch_size=args.finbert_batch_size,
                ),
            }
        )
    sentiment_df = pd.DataFrame(sentiment_rows)
    analysis_df = buyback_sample.merge(sentiment_df, on="transcriptid", how="left")

    qa_buyback_pairs_clean = pd.DataFrame()
    clarity_components = pd.DataFrame(
        columns=[
            "transcriptid",
            "specificity",
            "hedge_density",
            "modified_fog",
            "qa_relevance",
            "clarity_composite",
        ]
    )
    if not qa_pairs.empty:
        buyback_pattern = build_buyback_pattern()
        qa_pairs_flagged = flag_suspicious_qa_pairs(qa_pairs)
        qa_pairs_flagged["buyback_pair"] = (
            qa_pairs_flagged["question_text"].str.contains(buyback_pattern, na=False)
            | qa_pairs_flagged["response_text"].str.contains(buyback_pattern, na=False)
        )
        qa_buyback_pairs_clean = qa_pairs_flagged.loc[
            qa_pairs_flagged["buyback_pair"] & ~qa_pairs_flagged["is_suspicious"]
        ].head(args.max_clarity_pairs).copy()
        print(
            f"[sample] Clean buyback Q&A pairs selected for clarity: {len(qa_buyback_pairs_clean):,}.",
            flush=True,
        )

        if not qa_buyback_pairs_clean.empty:
            print("[sample] Computing clarity features...", flush=True)
            qa_buyback_pairs_clean["specificity"] = qa_buyback_pairs_clean["response_text"].map(compute_specificity)
            qa_buyback_pairs_clean["hedge_density"] = qa_buyback_pairs_clean["response_text"].map(compute_hedge_density)
            qa_buyback_pairs_clean["modified_fog"] = qa_buyback_pairs_clean["response_text"].map(compute_modified_fog)
            qa_buyback_pairs_clean["qa_relevance"] = [
                compute_qa_relevance(question, response, device=device)
                for question, response in zip(
                    qa_buyback_pairs_clean["question_text"],
                    qa_buyback_pairs_clean["response_text"],
                )
            ]
            clarity_components = (
                qa_buyback_pairs_clean.groupby("transcriptid")[
                    ["specificity", "hedge_density", "modified_fog", "qa_relevance"]
                ]
                .mean()
                .reset_index()
            )
            clarity_components["clarity_composite"] = compute_clarity_composite(
                clarity_components["specificity"],
                clarity_components["hedge_density"],
                clarity_components["modified_fog"],
                clarity_components["qa_relevance"],
            )

    analysis_df = analysis_df.merge(clarity_components, on="transcriptid", how="left")

    trend_surprise = compute_trend_revenue_surprise(analysis_df)
    ibes_placeholder = pd.DataFrame(columns=["transcriptid", "ibes_revenue_surprise"])
    analysis_df = merge_revenue_surprise(analysis_df, ibes_placeholder, trend_surprise)
    analysis_df["sentiment_bucket"] = bucket_sentiment(
        analysis_df["buyback_sentiment_mean"],
        groupby=analysis_df["calendar_quarter"],
    )
    analysis_df["clarity_bucket"] = bucket_clarity(
        analysis_df["clarity_composite"],
        groupby=analysis_df["calendar_quarter"],
    )
    analysis_df["revenue_bucket"] = bucket_revenue(analysis_df["revenue_surprise"])

    print("[sample] Running image-spec event study...", flush=True)
    event_results = run_event_study_from_wide_returns(
        analysis_df,
        estimation_window=IMAGE_SPEC_ESTIMATION_WINDOW,
        event_windows=IMAGE_SPEC_ROBUSTNESS_WINDOWS,
    )
    car_df = analysis_df[
        [
            "transcriptid",
            "permno",
            "call_date",
            "sentiment_bucket",
            "clarity_bucket",
            "revenue_bucket",
            "buyback_sentiment_mean",
            "clarity_composite",
            "revenue_surprise",
        ]
    ].copy().merge(event_results, on=["transcriptid", "permno", "call_date"], how="left")
    car_df["car"] = car_df["car_1_3"]

    matrix_results = create_sentiment_clarity_matrix(car_df, "sentiment_bucket", "clarity_bucket", "car")
    caar_results = compute_caar_by_bins(car_df, ["sentiment_bucket", "clarity_bucket", "revenue_bucket"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix_results.to_csv(args.output_matrix, index=False)

    summary = {
        "device": device,
        "transcript_rows_loaded": int(len(transcripts)),
        "buyback_calls": int(len(buyback_sample)),
        "component_path": str(component_path) if component_path is not None else None,
        "component_rows": int(len(component_rows)),
        "prepared_rows": int(len(prepared_df)),
        "qa_rows": int(len(qa_df)),
        "qa_pairs": int(len(qa_pairs)),
        "qa_pair_quality": qa_pair_quality,
        "clean_buyback_pairs_scored": int(len(qa_buyback_pairs_clean)),
        "clarity_transcripts": int(clarity_components["transcriptid"].nunique()) if not clarity_components.empty else 0,
        "sentiment_non_null": int(analysis_df["buyback_sentiment_mean"].notna().sum()),
        "clarity_non_null": int(analysis_df["clarity_composite"].notna().sum()),
        "revenue_non_null": int(analysis_df["revenue_surprise"].notna().sum()),
        "event_rows": int(len(event_results)),
        "car_1_3_non_null": int(event_results["car_1_3"].notna().sum()) if "car_1_3" in event_results.columns else 0,
        "car_1_5_non_null": int(event_results["car_1_5"].notna().sum()) if "car_1_5" in event_results.columns else 0,
        "matrix_rows": int(len(matrix_results)),
        "caar_rows": int(len(caar_results)),
    }

    args.output_json.write_text(json.dumps(summary, indent=2))
    print("[sample] Smoke test complete.", flush=True)
    print(json.dumps(summary, indent=2))
    print(f"Saved sample matrix to {args.output_matrix}")
    print(f"Saved sample summary to {args.output_json}")
    print(f"Project outputs live under {paths.outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
