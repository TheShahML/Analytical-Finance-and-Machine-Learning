# Buyback Sentiment and Clarity Research Repository

## Overview

This repository supports an Analytical Finance and Machine Learning course project focused on a new research question:

> Can the sentiment and clarity of executive language in earnings calls predict abnormal stock returns following a share buyback announcement?

The repo now treats `FINAL.csv` as the canonical cleaned transcript-level input and uses a true WRDS component-level export for prepared remarks / Q&A / speaker splits when available.

## Current Workflow

The project is organized around a buyback-first workflow:

1. Load the cleaned transcript panel from `FINAL.csv`.
2. Identify buyback-related calls and extract buyback text.
3. Load true WRDS transcript components for Q&A and speaker splits.
4. Score sentiment with FinBERT.
5. Compute buyback clarity on filtered Q&A pairs.
6. Run the event study using the image-spec windows from the current project design.
7. Build sentiment × clarity result tables.

## Data Inputs

Canonical transcript-level input:

- `data/FINAL.csv`

Preferred true component-level input:

- `data/interim/wrds_transcript_components.csv`

Heuristic fallback component input:

- `data/interim/transcript_components.csv`

Notes:

- `FINAL.csv` is treated as already cleaned.
- older pull / cleaning workflows are still present only as WIP provenance or legacy support
- the repo does not version raw data or large generated outputs in git

## Notebook Order

Main workflow (flat `notebooks/`; run order matches numeric prefixes):

1. [notebooks/00_transcript_and_revenue_data_pull.ipynb](notebooks/00_transcript_and_revenue_data_pull.ipynb) — WIP provenance for WRDS/raw pulls. Skip if `FINAL.csv` already exists locally.
2. [notebooks/01_data_audit_and_validation.ipynb](notebooks/01_data_audit_and_validation.ipynb) — Audit and validate the transcript base; writes processed/interim artifacts.
3. [notebooks/02_initial_eda.ipynb](notebooks/02_initial_eda.ipynb) — General EDA (keywords, topics) on the audited panel.
4. [notebooks/03_buyback_sentiment_clarity.ipynb](notebooks/03_buyback_sentiment_clarity.ipynb) — **Master pipeline:** buyback detection, FinBERT sentiment, Q&A clarity, revenue surprise, event study, sentiment × clarity tables (`FINAL.csv` path).
5. [notebooks/04_buyback_announcement_classification_ollama.ipynb](notebooks/04_buyback_announcement_classification_ollama.ipynb) — LLM classification of buyback excerpts (Ollama); run **before** the EDA notebooks that summarize those labels.
6. [notebooks/05_sentiment_clarity_eda_outputs.ipynb](notebooks/05_sentiment_clarity_eda_outputs.ipynb) — Faster sentiment × clarity EDA and presentation outputs (e.g. under `outputs/eda/`).
7. [notebooks/06_classification_eda_outputs.ipynb](notebooks/06_classification_eda_outputs.ipynb) — Scaffold for EDA on classification outputs from notebook04.

Archive / exploratory:

- [notebooks/90_keyword_and_theme_exploration.ipynb](notebooks/90_keyword_and_theme_exploration.ipynb)
- [notebooks/91_topic_modeling.ipynb](notebooks/91_topic_modeling.ipynb)
- [notebooks/98_panel_tests_legacy.ipynb](notebooks/98_panel_tests_legacy.ipynb) — legacy panel design; `src/finance/panel_regression.py` was removed (recover from git if needed).

See [notebooks/README.md](notebooks/README.md) for the same list.

## Key Modules

Data:

- `src/data/load_transcripts.py`
- `src/data/load_transcript_components.py`
- `src/data/wrds_transcript_components.py`
- `src/data/buyback_events.py`
- `src/data/qa_split.py`
- `src/data/revenue_surprise.py`

Features:

- `src/features/finbert_sentiment.py`
- `src/features/clarity.py`

Finance:

- `src/finance/event_study.py`
- `src/analysis/binning.py`

## Event Study Design

The active event-study implementation follows the current slide/image design:

- estimation window: `[-15, -3]`
- post-event windows: `[+1, +3]` and `[+1, +5]`
- primary expected-return model: mean model
- current implementation uses the wide `ret_t...` columns already present in `FINAL.csv`

## Q&A / Clarity Status

The repository now supports two component sources:

- true WRDS component rows from `wrds_transcript_components.csv`
- heuristic fallback rows from `transcript_components.csv`

For clarity work, the WRDS component file should be preferred. The clarity pipeline should still filter suspicious analyst/executive pairs before scoring.

## Setup

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### WRDS Component Pull

If you need to build the true component-level transcript file:

```bash
python3 scripts/pull_wrds_transcript_components.py --batch-size 200
```

Other helper scripts in `scripts/`: `build_heuristic_transcript_components.py` (heuristic Q&A rows), `compare_buyback_samples.py` (keyword vs event samples), `run_buyback_sample.py` (CLI smoke run of the buyback pipeline).

PowerShell users should set `WRDS_USERNAME` with:

```powershell
$env:WRDS_USERNAME = "your_wrds_username"
```

### Tests

```bash
pytest
```

## Current Practical Status

Ready now:

- transcript-level buyback detection
- WRDS-based Q&A speaker split
- FinBERT sentiment pipeline
- event study with the current design
- result-table scaffolding

Use with care:

- buyback clarity should use filtered Q&A pairs only

Archived / not current:

- panel-regression notebook flow
- older exploratory notebook order
