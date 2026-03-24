"""Interpretable keyword, concept, and bag-of-words helpers for Task 2 EDA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "r_and_d": ["r&d", "research and development"],
    "dividends": ["dividend", "dividends", "buyback", "buybacks", "share repurchase"],
    "guidance": ["guidance", "outlook", "forecast"],
    "layoffs": ["layoff", "layoffs", "restructuring", "headcount reduction"],
    "hiring": ["hire", "hiring", "recruiting", "talent acquisition"],
    "acquisitions": ["acquisition", "acquisitions", "acquire", "merger", "m&a", "expansion"],
}


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for simple concept matching."""

    return re.sub(r"\s+", " ", str(text).lower()).strip()


def build_keyword_pattern(terms: Sequence[str]) -> str:
    """Build one regex pattern for a concept's exact words or phrases."""

    escaped_terms = [re.escape(term.lower()) for term in terms]
    return rf"(?<!\w)(?:{'|'.join(escaped_terms)})(?!\w)"


def count_keyword_matches(text: str, terms: Sequence[str]) -> int:
    """Count exact keyword or phrase occurrences using word boundaries."""

    normalized = normalize_text(text)
    return _count_keyword_matches_normalized(normalized, terms)


def _count_keyword_matches_normalized(normalized_text: str, terms: Sequence[str]) -> int:
    """Count exact keyword or phrase occurrences on already-normalized text."""

    pattern = build_keyword_pattern(terms)
    return len(re.findall(pattern, normalized_text))


def count_keyword_mentions(
    df: pd.DataFrame,
    text_col: str,
    keywords: Sequence[str],
    *,
    id_col: str | None = None,
    output_col: str = "keyword_count",
) -> pd.DataFrame:
    """Count mentions of one keyword set for each transcript.

    Matching is exact whole-term / whole-phrase matching after lowercasing and
    whitespace normalization. This keeps the method transparent for early EDA.
    """

    if text_col not in df.columns:
        raise KeyError(f"Expected text column `{text_col}`.")

    result = df[[id_col]].copy() if id_col and id_col in df.columns else pd.DataFrame(index=df.index)
    normalized_text = df[text_col].fillna("").astype(str).map(normalize_text)
    result[output_col] = normalized_text.str.count(build_keyword_pattern(keywords))
    return result


def build_keyword_feature_table(
    df: pd.DataFrame,
    keyword_sets: Mapping[str, Sequence[str]],
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Create transcript-level concept counts for several keyword sets."""

    if text_column not in df.columns or id_column not in df.columns:
        raise KeyError(f"Expected columns `{id_column}` and `{text_column}` in transcript data.")

    result = df[[id_column]].copy()
    normalized_text = df[text_column].fillna("").astype(str).map(normalize_text)

    for theme_name, terms in keyword_sets.items():
        result[f"{theme_name}_keyword_count"] = normalized_text.str.count(build_keyword_pattern(terms))

    return result


def keyword_frequency_summary(
    keyword_feature_table: pd.DataFrame,
    *,
    id_col: str = "transcriptid",
) -> pd.DataFrame:
    """Summarize total mentions and document prevalence for each keyword set."""

    value_columns = [column for column in keyword_feature_table.columns if column != id_col]
    summary_rows: list[dict[str, object]] = []

    for column in value_columns:
        counts = pd.to_numeric(keyword_feature_table[column], errors="coerce").fillna(0)
        summary_rows.append(
            {
                "keyword_theme": column.replace("_keyword_count", ""),
                "total_mentions": int(counts.sum()),
                "transcripts_with_mentions": int((counts > 0).sum()),
                "share_of_transcripts": float((counts > 0).mean()),
                "mean_mentions_per_transcript": float(counts.mean()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["transcripts_with_mentions", "total_mentions"],
        ascending=False,
    ).reset_index(drop=True)


def keyword_frequency_by_year(
    df: pd.DataFrame,
    keyword_feature_table: pd.DataFrame,
    *,
    date_col: str = "call_date",
    id_col: str = "transcriptid",
) -> pd.DataFrame:
    """Aggregate transcript-level keyword counts to the year level."""

    if date_col not in df.columns or id_col not in df.columns:
        raise KeyError(f"Expected columns `{id_col}` and `{date_col}` in transcript data.")

    merged = df[[id_col, date_col]].merge(keyword_feature_table, on=id_col, how="inner")
    merged["year"] = pd.to_datetime(merged[date_col], errors="coerce").dt.year
    merged = merged.dropna(subset=["year"]).copy()
    merged["year"] = merged["year"].astype(int)

    rows: list[dict[str, object]] = []
    value_columns = [column for column in keyword_feature_table.columns if column != id_col]

    for year, group in merged.groupby("year", sort=True):
        transcript_count = len(group)
        for column in value_columns:
            counts = pd.to_numeric(group[column], errors="coerce").fillna(0)
            rows.append(
                {
                    "year": int(year),
                    "keyword_theme": column.replace("_keyword_count", ""),
                    "transcript_count": transcript_count,
                    "total_mentions": int(counts.sum()),
                    "transcripts_with_mentions": int((counts > 0).sum()),
                    "share_of_transcripts": float((counts > 0).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["keyword_theme", "year"]).reset_index(drop=True)


def keyword_frequency_by_firm(
    df: pd.DataFrame,
    keyword_feature_table: pd.DataFrame,
    *,
    firm_col: str = "companyname",
    id_col: str = "transcriptid",
    keyword_theme: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Summarize one keyword theme by firm for high-level descriptive review."""

    if firm_col not in df.columns or id_col not in df.columns:
        raise KeyError(f"Expected columns `{id_col}` and `{firm_col}` in transcript data.")

    keyword_column = (
        keyword_theme
        if keyword_theme in keyword_feature_table.columns
        else f"{keyword_theme}_keyword_count"
    )
    if keyword_column not in keyword_feature_table.columns:
        raise KeyError(f"Could not find keyword column `{keyword_column}`.")

    merged = df[[id_col, firm_col]].merge(
        keyword_feature_table[[id_col, keyword_column]],
        on=id_col,
        how="inner",
    )
    counts = pd.to_numeric(merged[keyword_column], errors="coerce").fillna(0)
    merged = merged.assign(_keyword_count=counts)

    summary = (
        merged.groupby(firm_col, dropna=False)
        .agg(
            transcript_count=(firm_col, "size"),
            total_mentions=("_keyword_count", "sum"),
            transcripts_with_mentions=("_keyword_count", lambda series: int((series > 0).sum())),
        )
        .reset_index()
    )
    summary["share_of_transcripts"] = (
        summary["transcripts_with_mentions"] / summary["transcript_count"].clip(lower=1)
    )

    return summary.sort_values(
        ["transcripts_with_mentions", "total_mentions", "transcript_count"],
        ascending=False,
    ).head(top_n).reset_index(drop=True)


def build_term_frequency_table(
    df: pd.DataFrame,
    *,
    text_col: str = "full_transcript_text",
    max_features: int = 50,
    min_df: int | float = 25,
    max_df: int | float = 0.9,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | Sequence[str] | None = "english",
) -> pd.DataFrame:
    """Build an interpretable bag-of-words frequency table with document counts."""

    if text_col not in df.columns:
        raise KeyError(f"Expected text column `{text_col}`.")

    texts = df[text_col].fillna("").astype(str).str.strip()
    texts = texts.loc[texts != ""]
    if texts.empty:
        return pd.DataFrame(columns=["term", "total_count", "document_frequency", "share_of_documents"])

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    total_count = matrix.sum(axis=0).A1
    document_frequency = (matrix > 0).sum(axis=0).A1

    result = pd.DataFrame(
        {
            "term": terms,
            "total_count": total_count.astype(int),
            "document_frequency": document_frequency.astype(int),
            "share_of_documents": document_frequency / max(len(texts), 1),
        }
    )
    return result.sort_values(["document_frequency", "total_count"], ascending=False).reset_index(
        drop=True
    )
