"""Heuristic component extraction from full transcript text.

This is a fallback for cases where WRDS/CIQ component-level transcript rows
are not available. It infers prepared-remarks vs Q&A blocks and a rough
analyst-vs-executive split from paragraph structure and common conference-call
phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd


PREPARED_REMARKS_TYPE = 2
QA_TYPE = 3
EXECUTIVE_SPEAKER_TYPE = 1
ANALYST_SPEAKER_TYPE = 3
OPERATOR_SPEAKER_TYPE = 0

QA_START_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\b(?:now|we will now|we'?ll now|let'?s now) begin (?:the )?question(?:-and-| and )answer session\b",
        r"\bopen (?:the )?(?:lines|call) (?:up )?for (?:your )?questions\b",
        r"\bwe'?d like to open (?:the lines|it) up for (?:your )?questions\b",
    ]
]

QUESTION_PROMPT_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\bour next question comes from\b",
        r"\byour next question comes from\b",
        r"\byour first question comes from\b",
        r"\bwe'?ll take our next question from\b",
        r"\btoday'?s first question comes from\b",
        r"\bfinal question comes from\b",
        r"\boperator instructions\b",
    ]
]

EXECUTIVE_RESPONSE_PREFIXES = (
    "yes",
    "yeah",
    "yep",
    "well",
    "thank you",
    "thanks",
    "let me",
    "i think",
    "i would say",
    "look,",
    "look ",
    "first,",
    "first ",
    "john,",
    "mark,",
    "rich,",
    "luke,",
)

ANALYST_QUESTION_PREFIXES = (
    "can you",
    "could you",
    "what",
    "how",
    "when",
    "why",
    "do you",
    "did you",
    "is there",
    "are there",
    "just on",
    "just a",
    "and then",
    "my question",
    "one more question",
)


@dataclass
class ParagraphClassification:
    component_type_id: int
    speaker_type_id: int
    speaker_name: str | None = None


def split_transcript_paragraphs(text: object) -> list[str]:
    """Split a transcript into cleaned paragraph-like segments."""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = _insert_paragraph_breaks(normalized)
    parts = re.split(r"\n\s*\n+", normalized)
    cleaned = [re.sub(r"\s+", " ", part).strip() for part in parts]
    return [part for part in cleaned if part]


def _insert_paragraph_breaks(text: str) -> str:
    """Insert paragraph breaks around inline speaker labels and Q&A markers."""

    normalized = text

    # Split inline HTML-style speaker labels into their own blocks.
    normalized = re.sub(
        r"\s*(<strong>\s*[A-Za-z][A-Za-z .&,'/-]{0,80}\s*</strong>)\s*",
        r"\n\n\1 ",
        normalized,
        flags=re.IGNORECASE,
    )

    marker_patterns = [
        r"(?:now|we will now|we'?ll now|let'?s now) begin (?:the )?question(?:-and-| and )answer session",
        r"open (?:the )?(?:lines|call) (?:up )?for (?:your )?questions",
        r"we'?d like to open (?:the lines|it) up for (?:your )?questions",
        r"our next question comes from",
        r"your next question comes from",
        r"your first question comes from",
        r"today'?s first question comes from",
        r"we'?ll take our next question from",
        r"looks like our first question comes from",
        r"final question comes from",
    ]

    for marker_pattern in marker_patterns:
        normalized = re.sub(
            rf"(?<!\n)\s+({marker_pattern})\b",
            r"\n\n\1",
            normalized,
            flags=re.IGNORECASE,
        )

    prompt_sentence_pattern = (
        r"((?:our next question comes from|your next question comes from|your first question comes from|"
        r"today'?s first question comes from|we'?ll take our next question from|"
        r"looks like our first question comes from|final question comes from)"
        r"[^.?!]{0,300}[.?!])\s+"
    )
    normalized = re.sub(
        prompt_sentence_pattern,
        r"\1\n\n",
        normalized,
        flags=re.IGNORECASE,
    )

    return normalized


def _matches_any(paragraph: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(paragraph) for pattern in patterns)


def _extract_analyst_name_from_prompt(paragraph: str) -> str | None:
    match = re.search(
        r"(?:line|question) comes from the line of ([A-Za-z .'-]+?)(?: with | from |$)",
        paragraph,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def _looks_like_analyst_question(paragraph: str) -> bool:
    lower = paragraph.lower().strip()
    if not lower:
        return False
    if any(lower.startswith(prefix) for prefix in EXECUTIVE_RESPONSE_PREFIXES):
        return False
    if any(lower.startswith(prefix) for prefix in ANALYST_QUESTION_PREFIXES):
        return True
    if "?" in paragraph and not lower.startswith(EXECUTIVE_RESPONSE_PREFIXES):
        return True
    return False


def classify_paragraphs(paragraphs: list[str]) -> list[ParagraphClassification]:
    """Classify transcript paragraphs into prepared remarks and heuristic Q&A roles."""

    classifications: list[ParagraphClassification] = []
    in_qa = False
    expecting_analyst = False
    pending_analyst_name: str | None = None

    for paragraph in paragraphs:
        if _matches_any(paragraph, QA_START_PATTERNS) or _matches_any(paragraph, QUESTION_PROMPT_PATTERNS):
            in_qa = True

        if not in_qa:
            classifications.append(
                ParagraphClassification(
                    component_type_id=PREPARED_REMARKS_TYPE,
                    speaker_type_id=EXECUTIVE_SPEAKER_TYPE,
                )
            )
            continue

        if _matches_any(paragraph, QUESTION_PROMPT_PATTERNS):
            pending_analyst_name = _extract_analyst_name_from_prompt(paragraph)
            expecting_analyst = True
            classifications.append(
                ParagraphClassification(
                    component_type_id=QA_TYPE,
                    speaker_type_id=OPERATOR_SPEAKER_TYPE,
                )
            )
            continue

        if expecting_analyst or _looks_like_analyst_question(paragraph):
            classifications.append(
                ParagraphClassification(
                    component_type_id=QA_TYPE,
                    speaker_type_id=ANALYST_SPEAKER_TYPE,
                    speaker_name=pending_analyst_name,
                )
            )
            expecting_analyst = False
            pending_analyst_name = None
            continue

        classifications.append(
            ParagraphClassification(
                component_type_id=QA_TYPE,
                speaker_type_id=EXECUTIVE_SPEAKER_TYPE,
            )
        )

    return classifications


def extract_transcript_components_from_text(
    transcript_text: object,
    *,
    transcript_id: object | None = None,
) -> pd.DataFrame:
    """Create heuristic component rows from one full transcript."""

    paragraphs = split_transcript_paragraphs(transcript_text)
    classifications = classify_paragraphs(paragraphs)
    rows: list[dict[str, Any]] = []

    for order, (paragraph, classification) in enumerate(zip(paragraphs, classifications), start=1):
        rows.append(
            {
                "transcriptid": transcript_id,
                "componentorder": order,
                "componenttext": paragraph,
                "transcriptcomponenttypeid": classification.component_type_id,
                "speakertypeid": classification.speaker_type_id,
                "speakername": classification.speaker_name,
                "component_source": "heuristic",
            }
        )

    return pd.DataFrame(rows)


def build_component_dataset_from_transcripts(
    transcripts_df: pd.DataFrame,
    *,
    transcript_id_col: str = "transcriptid",
    text_col: str = "full_transcript_text",
    extra_columns: tuple[str, ...] = ("companyid", "companyname", "ticker", "call_date"),
) -> pd.DataFrame:
    """Build a heuristic component dataset from transcript-level rows."""

    if transcript_id_col not in transcripts_df.columns or text_col not in transcripts_df.columns:
        raise KeyError(f"Expected columns `{transcript_id_col}` and `{text_col}`.")

    frames: list[pd.DataFrame] = []
    for row in transcripts_df.itertuples(index=False):
        transcript_id = getattr(row, transcript_id_col)
        transcript_text = getattr(row, text_col)
        component_df = extract_transcript_components_from_text(
            transcript_text,
            transcript_id=transcript_id,
        )
        for column in extra_columns:
            if column in transcripts_df.columns:
                component_df[column] = getattr(row, column)
        frames.append(component_df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "transcriptid",
                "componentorder",
                "componenttext",
                "transcriptcomponenttypeid",
                "speakertypeid",
                "speakername",
                "component_source",
                *extra_columns,
            ]
        )

    return pd.concat(frames, ignore_index=True)
