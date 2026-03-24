"""Panel-regression scaffolding for later finance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class PanelRegressionSpec:
    """Specification container for later panel regressions."""

    outcome: str
    feature_columns: Sequence[str]
    control_columns: Sequence[str] = field(default_factory=tuple)
    entity_effects: bool = False
    time_effects: bool = True


def build_regression_formula(spec: PanelRegressionSpec) -> str:
    """Construct a simple formula string from a panel-regression specification."""

    rhs_terms = list(spec.feature_columns) + list(spec.control_columns)
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    return f"{spec.outcome} ~ {rhs}"


def run_panel_regression(
    df: pd.DataFrame,
    spec: PanelRegressionSpec,
    entity_id: str = "permno",
    time_id: str = "call_date",
):
    """Placeholder for later panel-regression estimation."""

    raise NotImplementedError("Panel regression is reserved for a later implementation pass.")
