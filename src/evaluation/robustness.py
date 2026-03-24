"""Planning helpers for future robustness checks."""

from __future__ import annotations


def default_robustness_checks() -> list[str]:
    """Return a starter list of robustness checks to consider later."""

    return [
        "separate prepared remarks from Q&A",
        "exclude transcripts with unusually high boilerplate language",
        "winsorize extreme transcript features",
        "cluster standard errors at the firm level",
        "test alternative event windows",
        "compare dictionary-based scores to model-assisted scores",
    ]
