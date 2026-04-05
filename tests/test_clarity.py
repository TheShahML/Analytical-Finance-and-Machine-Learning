"""Tests for clarity-feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.clarity import (
    FINANCIAL_VOCAB_EXCLUSIONS,
    compute_clarity_composite,
    compute_hedge_density,
    compute_modified_fog,
    compute_specificity,
    load_lm_dictionary,
)


def test_compute_specificity_scores_all_four_detail_types() -> None:
    text = (
        "We approved a $3 billion repurchase of 25 million shares "
        "over the next 12 months funded by free cash flow."
    )

    assert compute_specificity(text) == 4


def test_load_lm_dictionary_reads_uncertainty_and_weak_modal_columns(tmp_path) -> None:
    dictionary_path = tmp_path / "lm_dictionary.csv"
    pd.DataFrame(
        {
            "Word": ["uncertain", "may", "certain"],
            "Uncertainty": [1, 0, 0],
            "Weak Modal": [0, 1, 0],
        }
    ).to_csv(dictionary_path, index=False)

    uncertainty, weak_modal = load_lm_dictionary(dictionary_path)

    assert "uncertain" in uncertainty
    assert "may" in weak_modal


def test_compute_hedge_density_uses_dictionary_matches(tmp_path) -> None:
    dictionary_path = tmp_path / "lm_dictionary.csv"
    pd.DataFrame(
        {
            "Word": ["uncertain", "may", "board"],
            "Uncertainty": [1, 0, 0],
            "Weak Modal": [0, 1, 0],
        }
    ).to_csv(dictionary_path, index=False)

    density = compute_hedge_density("We may remain uncertain.", lm_dict_path=dictionary_path)

    assert round(density, 2) == 0.5


def test_compute_modified_fog_excludes_common_financial_vocabulary() -> None:
    text = "Management administration infrastructure clarity simplicity."

    fog_with_exclusions = compute_modified_fog(text, exclusion_set=FINANCIAL_VOCAB_EXCLUSIONS)
    fog_without_exclusions = compute_modified_fog(text, exclusion_set=set())

    assert fog_with_exclusions < fog_without_exclusions


def test_compute_clarity_composite_returns_centered_array() -> None:
    composite = compute_clarity_composite(
        specificity=[1, 2, 3],
        hedge_density=[0.4, 0.2, 0.1],
        fog=[18.0, 14.0, 10.0],
        qa_relevance=[0.2, 0.5, 0.8],
    )

    assert isinstance(composite, np.ndarray)
    assert len(composite) == 3
    assert np.isclose(composite.mean(), 0.0)
