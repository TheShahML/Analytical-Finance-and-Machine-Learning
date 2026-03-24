"""Lightweight topic-discovery utilities for Task 2 exploratory analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.features.text_preprocessing import prepare_corpus


def prepare_documents_for_topic_modeling(
    df: pd.DataFrame,
    *,
    text_column: str = "full_transcript_text",
    id_column: str = "transcriptid",
    extra_columns: Sequence[str] | None = None,
    output_text_column: str = "clean_text",
    min_characters: int = 250,
) -> pd.DataFrame:
    """Return a lightly cleaned transcript subset for topic exploration."""

    required = [id_column, text_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Expected columns for topic modeling: {missing}")

    columns = [id_column, text_column]
    if extra_columns:
        columns.extend([column for column in extra_columns if column in df.columns])

    documents = prepare_corpus(
        df[columns].copy(),
        text_col=text_column,
        output_col=output_text_column,
        min_characters=min_characters,
    )
    return documents.reset_index(drop=True)


def sample_topic_documents(
    documents: pd.DataFrame,
    *,
    max_documents: int = 5000,
    date_column: str | None = "call_date",
    random_state: int = 42,
) -> pd.DataFrame:
    """Downsample documents for faster exploratory topic fitting.

    If a usable date column is present, sampling is approximately year-balanced.
    """

    if len(documents) <= max_documents:
        return documents.reset_index(drop=True)

    if date_column and date_column in documents.columns:
        sampled_groups = []
        dates = pd.to_datetime(documents[date_column], errors="coerce")
        docs = documents.assign(_year=dates.dt.year, _source_index=documents.index)
        valid = docs.loc[docs["_year"].notna()].copy()
        if not valid.empty:
            per_year = max(max_documents // valid["_year"].nunique(), 1)
            for _, group in valid.groupby("_year", sort=True):
                sampled_groups.append(group.sample(n=min(len(group), per_year), random_state=random_state))
            sampled = pd.concat(sampled_groups, ignore_index=True)
            if len(sampled) < max_documents:
                used_source_indices = sampled["_source_index"].tolist()
                remaining = docs.loc[~docs.index.isin(used_source_indices)]
                extra_n = min(max_documents - len(sampled), len(remaining))
                if extra_n > 0:
                    sampled = pd.concat(
                        [sampled, remaining.sample(extra_n, random_state=random_state)],
                        ignore_index=True,
                    )
            sampled = sampled.head(max_documents).drop(
                columns=["_year", "_source_index"],
                errors="ignore",
            )
            return sampled.reset_index(drop=True)

    return documents.sample(n=max_documents, random_state=random_state).reset_index(drop=True)


def fit_lda_topic_model(
    documents: pd.DataFrame,
    *,
    text_column: str = "clean_text",
    n_topics: int = 6,
    max_features: int = 1500,
    min_df: int | float = 20,
    max_df: int | float = 0.85,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | Sequence[str] | None = "english",
    random_state: int = 42,
    max_iter: int = 10,
) -> dict[str, Any]:
    """Fit a simple sklearn LDA model for exploratory topic discovery."""

    if text_column not in documents.columns:
        raise KeyError(f"Expected text column `{text_column}`.")

    texts = documents[text_column].fillna("").astype(str).str.strip()
    texts = texts.loc[texts != ""]
    if texts.empty:
        raise ValueError("No non-empty documents available for topic modeling.")

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    document_term_matrix = vectorizer.fit_transform(texts)

    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=max_iter,
    )
    document_topic_matrix = model.fit_transform(document_term_matrix)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "document_term_matrix": document_term_matrix,
        "document_topic_matrix": document_topic_matrix,
        "documents_used": documents.loc[texts.index].reset_index(drop=True),
    }


def extract_top_words_per_topic(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    *,
    n_top_words: int = 12,
) -> pd.DataFrame:
    """Return the highest-weighted words or phrases for each topic."""

    vocabulary = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []

    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[::-1][:n_top_words]
        top_terms = [vocabulary[index] for index in top_indices]
        rows.append(
            {
                "topic_id": int(topic_idx),
                "top_terms": ", ".join(top_terms),
            }
        )

    return pd.DataFrame(rows)


def assign_dominant_topic(
    documents: pd.DataFrame,
    document_topic_matrix,
    *,
    id_column: str = "transcriptid",
) -> pd.DataFrame:
    """Assign the highest-probability topic to each document."""

    if len(documents) != len(document_topic_matrix):
        raise ValueError("Document-topic matrix length must match the documents DataFrame.")

    topic_scores = pd.DataFrame(document_topic_matrix)
    dominant_topic = topic_scores.idxmax(axis=1).astype(int)
    dominant_topic_share = topic_scores.max(axis=1)

    result = documents.copy().reset_index(drop=True)
    result["dominant_topic"] = dominant_topic
    result["dominant_topic_share"] = dominant_topic_share

    if id_column in result.columns:
        columns = [id_column] + [column for column in result.columns if column != id_column]
        result = result.loc[:, columns]

    return result


def summarize_topic_prevalence(
    topic_assignments: pd.DataFrame,
    *,
    topic_column: str = "dominant_topic",
    date_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarize overall and optionally year-level topic prevalence."""

    if topic_column not in topic_assignments.columns:
        raise KeyError(f"Expected topic column `{topic_column}`.")

    overall = (
        topic_assignments[topic_column]
        .value_counts()
        .sort_index()
        .rename_axis("topic_id")
        .reset_index(name="document_count")
    )
    overall["share_of_documents"] = overall["document_count"] / max(len(topic_assignments), 1)

    by_year = pd.DataFrame(columns=["year", "topic_id", "document_count", "share_of_year_documents"])
    if date_column and date_column in topic_assignments.columns:
        dates = pd.to_datetime(topic_assignments[date_column], errors="coerce")
        with_year = topic_assignments.assign(year=dates.dt.year).dropna(subset=["year"]).copy()
        with_year["year"] = with_year["year"].astype(int)
        if not with_year.empty:
            by_year = (
                with_year.groupby(["year", topic_column])
                .size()
                .rename("document_count")
                .reset_index()
                .rename(columns={topic_column: "topic_id"})
            )
            totals = by_year.groupby("year")["document_count"].transform("sum")
            by_year["share_of_year_documents"] = by_year["document_count"] / totals.clip(lower=1)

    return {"overall": overall, "by_year": by_year}


def select_topic_examples(
    topic_assignments: pd.DataFrame,
    *,
    text_column: str = "full_transcript_text",
    topic_column: str = "dominant_topic",
    score_column: str = "dominant_topic_share",
    id_column: str = "transcriptid",
    n_examples_per_topic: int = 2,
    preview_characters: int = 180,
) -> pd.DataFrame:
    """Return high-confidence example snippets for each dominant topic."""

    required = [topic_column, score_column]
    missing = [column for column in required if column not in topic_assignments.columns]
    if missing:
        raise KeyError(f"Expected topic-assignment columns: {missing}")

    working = topic_assignments.copy()
    if text_column in working.columns:
        working["text_preview"] = (
            working[text_column].fillna("").astype(str).str.slice(0, preview_characters)
        )

    working = working.sort_values([topic_column, score_column], ascending=[True, False])
    grouped = working.groupby(topic_column, sort=True).head(n_examples_per_topic).reset_index(drop=True)

    columns = [column for column in [topic_column, score_column, id_column, "ticker", "companyname", "call_date", "text_preview"] if column in grouped.columns]
    return grouped.loc[:, columns]
