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

The notebooks are now split into a main workflow and an archive track.

Main workflow:

1. [00_transcript_and_revenue_data_pull.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/00_transcript_and_revenue_data_pull.ipynb)
   WIP provenance notebook for WRDS pulls. Not required if `FINAL.csv` already exists locally.
2. [01_data_audit_and_validation.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/01_data_audit_and_validation.ipynb)
   Validates and documents the cleaned transcript base.
3. [02_initial_eda.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/02_initial_eda.ipynb)
   General EDA on the cleaned transcript panel.
4. [03_sentiment_baseline.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/03_sentiment_baseline.ipynb)
   Transcript-level and buyback-level sentiment baseline work.
5. [04_clarity_signal_prototyping.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/04_clarity_signal_prototyping.ipynb)
   Clarity-feature prototyping on buyback Q&A.
6. [05_event_study_baseline.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/05_event_study_baseline.ipynb)
   Event-study baseline using the current image-spec design.
7. [06_buyback_sentiment_clarity.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/06_buyback_sentiment_clarity.ipynb)
   Master notebook for the full buyback pipeline.

Archive / exploratory notebooks:

- [90_keyword_and_theme_exploration.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/90_keyword_and_theme_exploration.ipynb)
- [91_topic_modeling.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/91_topic_modeling.ipynb)
- [98_panel_tests_legacy.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical%20Finance%20and%20Machine%20Learning/notebooks/98_panel_tests_legacy.ipynb)

`98_panel_tests_legacy.ipynb` is intentionally archived because panel regression is no longer part of the active workflow.

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
- `src/features/embeddings.py`

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
