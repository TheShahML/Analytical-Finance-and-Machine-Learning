"""Prepared-remarks and Q&A splitting helpers for Capital IQ transcripts."""

from __future__ import annotations

from typing import Any

import pandas as pd


PREPARED_REMARKS_TYPE = 2
QA_TYPE = 3
EXECUTIVE_SPEAKER_TYPE = 1
ANALYST_SPEAKER_TYPE = 3
WRDS_PREPARED_COMPONENT_TYPES = {1, 2}
WRDS_QA_COMPONENT_TYPES = {3, 4, 7, 8}
WRDS_EXECUTIVE_SPEAKER_TYPES = {2}
WRDS_ANALYST_SPEAKER_TYPES = {3}


def split_prepared_qa(
    transcripts_df: pd.DataFrame,
    component_col: str = "transcriptcomponenttypeid",
    prepared_types: set[int] | None = None,
    qa_types: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split component-level transcript records into prepared remarks and Q&A."""

    if component_col not in transcripts_df.columns:
        raise KeyError(f"Expected transcript component column `{component_col}`.")

    component_codes = set(pd.to_numeric(transcripts_df[component_col], errors="coerce").dropna().astype(int))
    if prepared_types is None or qa_types is None:
        if {4, 7}.intersection(component_codes):
            prepared_types = WRDS_PREPARED_COMPONENT_TYPES
            qa_types = WRDS_QA_COMPONENT_TYPES
        else:
            prepared_types = {PREPARED_REMARKS_TYPE}
            qa_types = {QA_TYPE}

    component_series = pd.to_numeric(transcripts_df[component_col], errors="coerce")
    prepared = transcripts_df.loc[
        component_series.isin(prepared_types)
    ].copy()
    qa = transcripts_df.loc[
        component_series.isin(qa_types)
    ].copy()
    return prepared.reset_index(drop=True), qa.reset_index(drop=True)


def split_analyst_executive(
    qa_df: pd.DataFrame,
    speaker_col: str = "speakertypeid",
    analyst_types: set[int] | None = None,
    executive_types: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within the Q&A section, split analyst questions from executive responses."""

    if speaker_col not in qa_df.columns:
        raise KeyError(f"Expected speaker-type column `{speaker_col}`.")

    speaker_codes = pd.to_numeric(qa_df[speaker_col], errors="coerce")
    observed_codes = set(speaker_codes.dropna().astype(int))
    if analyst_types is None or executive_types is None:
        if 2 in observed_codes and 3 in observed_codes and 1 in observed_codes:
            analyst_types = WRDS_ANALYST_SPEAKER_TYPES
            executive_types = WRDS_EXECUTIVE_SPEAKER_TYPES
        else:
            analyst_types = {ANALYST_SPEAKER_TYPE}
            executive_types = {EXECUTIVE_SPEAKER_TYPE}

    analysts = qa_df.loc[speaker_codes.isin(analyst_types)].copy()
    executives = qa_df.loc[speaker_codes.isin(executive_types)].copy()
    return analysts.reset_index(drop=True), executives.reset_index(drop=True)


def pair_questions_responses(
    qa_df: pd.DataFrame,
    *,
    transcript_id_col: str = "transcriptid",
    speaker_col: str = "speakertypeid",
    text_col: str = "componenttext",
    sequence_col: str | None = None,
) -> pd.DataFrame:
    """Pair each analyst question with the immediately subsequent executive response."""

    if transcript_id_col not in qa_df.columns:
        raise KeyError(f"Expected transcript identifier column `{transcript_id_col}`.")
    if speaker_col not in qa_df.columns:
        raise KeyError(f"Expected speaker-type column `{speaker_col}`.")

    if text_col not in qa_df.columns:
        fallback_columns = ["text", "content", "full_transcript_text"]
        available = next((column for column in fallback_columns if column in qa_df.columns), None)
        if available is None:
            raise KeyError(
                f"Expected question/response text column `{text_col}` or one of {fallback_columns}."
            )
        text_col = available

    ordered = qa_df.copy()
    if sequence_col and sequence_col in ordered.columns:
        ordered = ordered.sort_values([transcript_id_col, sequence_col]).copy()
    else:
        ordered = ordered.reset_index().rename(columns={"index": "_qa_original_order"})
        ordered = ordered.sort_values([transcript_id_col, "_qa_original_order"]).copy()
        sequence_col = "_qa_original_order"

    speaker_codes = pd.to_numeric(ordered[speaker_col], errors="coerce")
    observed_codes = set(speaker_codes.dropna().astype(int))
    if 2 in observed_codes and 3 in observed_codes and 1 in observed_codes:
        analyst_types = WRDS_ANALYST_SPEAKER_TYPES
        executive_types = WRDS_EXECUTIVE_SPEAKER_TYPES
    else:
        analyst_types = {ANALYST_SPEAKER_TYPE}
        executive_types = {EXECUTIVE_SPEAKER_TYPE}

    pairs: list[dict[str, Any]] = []
    for transcript_id, group in ordered.groupby(transcript_id_col, sort=False):
        group = group.reset_index(drop=True)
        pending_question: dict[str, Any] | None = None

        for row in group.to_dict(orient="records"):
            speaker_type = pd.to_numeric(row.get(speaker_col), errors="coerce")
            if speaker_type in analyst_types:
                pending_question = row
                continue

            if speaker_type in executive_types and pending_question is not None:
                pairs.append(
                    {
                        transcript_id_col: transcript_id,
                        "question_order": pending_question.get(sequence_col),
                        "response_order": row.get(sequence_col),
                        "question_text": str(pending_question.get(text_col, "") or "").strip(),
                        "response_text": str(row.get(text_col, "") or "").strip(),
                    }
                )
                pending_question = None

    return pd.DataFrame(pairs)


def flag_suspicious_qa_pairs(
    pairs_df: pd.DataFrame,
    *,
    question_col: str = "question_text",
    response_col: str = "response_text",
    question_word_ceiling: int = 250,
    response_word_ceiling: int = 800,
) -> pd.DataFrame:
    """Attach simple quality flags for potentially bad heuristic Q&A pairs."""

    required = {question_col, response_col}
    missing = required.difference(pairs_df.columns)
    if missing:
        raise KeyError(f"Expected Q&A pair columns {sorted(required)}; missing {sorted(missing)}.")

    flagged = pairs_df.copy()
    question_text = flagged[question_col].fillna("").astype(str)
    response_text = flagged[response_col].fillna("").astype(str)

    flagged["question_word_count"] = question_text.str.split().str.len()
    flagged["response_word_count"] = response_text.str.split().str.len()
    flagged["question_contains_operator_tag"] = question_text.str.contains(
        r"<strong>\s*operator\s*</strong>|^operator\b",
        case=False,
        regex=True,
    )
    flagged["response_contains_operator_tag"] = response_text.str.contains(
        r"<strong>\s*operator\s*</strong>|^operator\b",
        case=False,
        regex=True,
    )
    flagged["question_contains_response_cue"] = question_text.str.contains(
        r"\b(?:i would say|let me|look,|look |thank you|thanks|well,|well )\b",
        case=False,
        regex=True,
    )
    flagged["question_too_long"] = flagged["question_word_count"] > question_word_ceiling
    flagged["response_too_long"] = flagged["response_word_count"] > response_word_ceiling
    flagged["is_suspicious"] = flagged[
        [
            "question_contains_operator_tag",
            "response_contains_operator_tag",
            "question_contains_response_cue",
            "question_too_long",
            "response_too_long",
        ]
    ].any(axis=1)
    return flagged


def summarize_qa_pair_quality(
    pairs_df: pd.DataFrame,
    *,
    question_col: str = "question_text",
    response_col: str = "response_text",
) -> dict[str, float | int]:
    """Return compact diagnostics for heuristic Q&A pair quality."""

    flagged = flag_suspicious_qa_pairs(
        pairs_df,
        question_col=question_col,
        response_col=response_col,
    )
    pair_count = int(len(flagged))
    suspicious_count = int(flagged["is_suspicious"].sum())

    return {
        "pair_count": pair_count,
        "suspicious_pairs": suspicious_count,
        "suspicious_share": float(suspicious_count / pair_count) if pair_count else 0.0,
        "median_question_words": float(flagged["question_word_count"].median()) if pair_count else 0.0,
        "median_response_words": float(flagged["response_word_count"].median()) if pair_count else 0.0,
    }


def validate_qa_split(
    transcripts_df: pd.DataFrame,
    sample_n: int = 50,
    *,
    component_col: str = "transcriptcomponenttypeid",
    speaker_col: str = "speakertypeid",
    random_state: int = 42,
) -> dict[str, object]:
    """Return summary diagnostics and a small audit sample for the Q&A split."""

    prepared, qa = split_prepared_qa(transcripts_df, component_col=component_col)
    analyst, executive = split_analyst_executive(qa, speaker_col=speaker_col)

    audit_columns = [
        column
        for column in [
            "transcriptid",
            component_col,
            speaker_col,
            "speakername",
            "componenttext",
            "text",
        ]
        if column in transcripts_df.columns
    ]
    audit_sample = qa.loc[:, audit_columns].sample(
        n=min(sample_n, len(qa)),
        random_state=random_state,
    ) if len(qa) else qa.loc[:, audit_columns]

    summary = {
        "row_count": int(len(transcripts_df)),
        "prepared_rows": int(len(prepared)),
        "qa_rows": int(len(qa)),
        "qa_share": float(len(qa) / len(transcripts_df)) if len(transcripts_df) else 0.0,
        "analyst_question_rows": int(len(analyst)),
        "executive_response_rows": int(len(executive)),
        "audit_sample": audit_sample.reset_index(drop=True),
    }
    return summary
