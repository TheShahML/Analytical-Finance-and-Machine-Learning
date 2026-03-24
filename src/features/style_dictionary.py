"""Seed dictionaries for interpretable communication-style prototyping."""

from __future__ import annotations


STYLE_DICTIONARY_SEEDS: dict[str, list[str]] = {
    "hedging": [
        "may",
        "might",
        "could",
        "possibly",
        "approximately",
        "expect",
        "believe",
        "anticipate",
        "uncertain",
    ],
    "vagueness": [
        "some",
        "several",
        "various",
        "certain",
        "relatively",
        "somewhat",
        "around",
        "roughly",
    ],
    "directness": [
        "specifically",
        "clearly",
        "exactly",
        "we will",
        "we did",
        "our plan",
        "our results",
    ],
    "transparency": [
        "disclose",
        "detail",
        "explain",
        "clarify",
        "breakdown",
        "data",
        "assumption",
    ],
    "attribution_accountability": [
        "we",
        "our decision",
        "management",
        "responsibility",
        "accountable",
        "ownership",
    ],
    "evasive_proxy": [
        "as you know",
        "at a high level",
        "broadly speaking",
        "not going to",
        "cannot comment",
        "hard to say",
    ],
}


def get_style_dictionary(signal_name: str) -> list[str]:
    """Return a seed dictionary for a named communication-style signal."""

    if signal_name not in STYLE_DICTIONARY_SEEDS:
        available = ", ".join(sorted(STYLE_DICTIONARY_SEEDS))
        raise KeyError(f"Unknown signal `{signal_name}`. Available signals: {available}")
    return STYLE_DICTIONARY_SEEDS[signal_name]


def get_style_signal_names() -> list[str]:
    """Return the available communication-style signal names."""

    return list(STYLE_DICTIONARY_SEEDS.keys())
