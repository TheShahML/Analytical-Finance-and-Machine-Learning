# Earnings Call Communication Style Research Repository

## Overview

This repository supports an Analytical Finance and Machine Learning course project built around earnings conference call transcripts. The immediate goal is to build a clean shared dataset and complete the first round of exploratory analysis. The longer-run goal is to study whether transcript-based communication style contains information beyond standard sentiment.

## Course Context

The project starts from an existing earnings call transcript corpus and a notebook for the original data pull. The professor's near-term priority is to establish a reliable shared dataset, validate it, document it, and complete initial EDA before stronger empirical claims are attempted.

## Project Objective

The objective is to build a modular research pipeline for transcript-based signals while keeping sentiment separate from communication style.

Candidate communication-style dimensions include:

- evasiveness or deflection
- vagueness
- hedging or uncertainty
- directness or specificity
- transparency
- accountability or attribution

The repository is designed to support multiple transcript-level signals rather than collapsing everything into a single score.

## Current Phase

The project is still in the foundation phase. Current emphasis:

- repository and workflow setup
- transcript pull extension to the 2010+ window
- shared dataset audit, validation, and cleaning
- exploratory EDA, keyword exploration, and topic discovery
- documentation of assumptions and reusable code paths

Not yet in scope:

- final empirical results
- advanced model selection
- strong claims about predictability or causal interpretation

## Three Workstreams

### Workstream 1: Shared Dataset Foundation

- consolidate the transcript corpus
- verify coverage and identifier quality
- audit missingness, duplicates, and filtering decisions
- complete initial EDA
- explore keywords, themes, and basic topic structure

### Workstream 2: Signal Exploration

- define transcript signal families
- keep sentiment distinct from communication-style measures
- prototype interpretable dictionary-based features first
- plan later model-assisted methods
- evaluate whether prepared remarks and Q&A should be analyzed separately

### Workstream 3: Research Question and Empirical Design

- frame the likely research question around incremental information beyond sentiment
- sketch event-study and panel or cross-sectional tests
- identify likely controls and robustness checks
- keep the design memo separate from any future claims document

## Current Known Assets

- canonical raw transcript target: `data/raw/earnings_calls_full_2010_onward_with_revenue.csv`
- current pull notebook: `notebooks/00_transcript_and_revenue_data_pull.ipynb`

The working transcript schema appears to include:

- transcript and company identifiers
- transcript text and metadata
- date fields related to the call and reporting cycle
- market linkage variables such as `permno`, `gvkey`, and `ibes_ticker`
- revenue and guidance-related fields

These fields still need to be formally documented in the data dictionary and validated in Task 01.

## Repository Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- .env.example
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- external/
|-- notebooks/
|-- src/
|   |-- config/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- finance/
|   |-- evaluation/
|   `-- utils/
|-- docs/
|-- outputs/
|   |-- figures/
|   |-- tables/
|   `-- reports/
`-- tests/
```

Practical intent:

- `data/` holds raw inputs, intermediate working files, and cleaned outputs
- `notebooks/` holds staged exploratory and prototype notebooks
- `src/` holds reusable loading, validation, feature, and finance logic
- `docs/` holds working design notes, templates, and project decisions
- `outputs/` holds generated figures, tables, and short reports
- `tests/` holds lightweight tests for reusable code interfaces

## Workflow

1. Pull or refresh the raw 2010+ transcript corpus.
2. Validate transcript identifiers, dates, coverage, duplicates, and missingness.
3. Build a documented cleaned transcript table.
4. Run initial EDA on counts, firms, length measures, keywords, and themes.
5. Prototype distinct transcript signals for sentiment and communication style.
6. Link transcript observations to finance data for baseline event-study and panel tests.

## Notebooks vs `src`

- notebooks for exploration, plotting, quick inspection, and staged experiments
- `src/` modules for reusable loading, validation, feature construction, and finance interfaces

Guideline:

- if logic will be reused, moved between notebooks, or tested, it belongs in `src/`
- if work is exploratory, visual, or one-off, it can begin in a notebook
- outputs from notebooks should be written to `outputs/`
- cleaned or intermediate datasets should be written to `data/interim/` or `data/processed/`

Practical note:

- the notebooks are set up to auto-detect the repository root and make `src/` importable inside Jupyter
- if you have an older kernel session open, restart the kernel after pulling notebook changes so the updated import bootstrap is used

## Setup

### Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure local paths

Copy `.env.example` to `.env` and update any paths needed for:

- transcript data
- returns data
- fundamentals data
- analyst data

### Data access

This repository does not include raw data or generated output files in git.

To use the repo, a teammate needs to do one of the following:

1. Run [00_transcript_and_revenue_data_pull.ipynb](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/notebooks/00_transcript_and_revenue_data_pull.ipynb) themselves, assuming they have the required WRDS access.
2. Manually place the required data files into the expected repo folders.

Expected file locations:

- raw transcript file: `data/raw/earnings_calls_full_2010_onward_with_revenue.csv`
- cleaned Task 01 file: `data/processed/CLEANED_earnings_calls_full_2010_onward_with_revenue.csv`

During the transition, the loader still supports fallback to the older legacy filename if the canonical raw file is not present.

### Run tests

```bash
pytest
```

## Current Known Limitations

- raw and processed data files are not versioned in git, so new users must create or place them locally
- the 2010+ raw pull still needs to be rerun and saved to the canonical `data/raw/` location
- the final canonical event date has not been chosen
- the transcript observation unit is not yet fully documented
- prepared remarks and Q&A have not yet been formally split or validated
- exploratory topic outputs should be treated as descriptive, not final constructs
- no empirical results in this repository should be treated as final at this stage

## Immediate Next Steps

- rerun Task 00 to build the 2010+ raw transcript file
- rerun Task 01 to rebuild the cleaned shared dataset from 2010 onward
- refresh Task 02 outputs on the updated cleaned corpus
- complete the first version of the data dictionary
- decide the first sentiment baseline and first interpretable style prototypes
- define how event dates and return windows will be handled later in the finance design

## Core Documents

- [Professor Requirements](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/docs/01_professor_requirements.md)
- [Signal Exploration Plan](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/docs/02_signal_exploration_plan.md)
- [Research Question Design](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/docs/03_research_question_design.md)
- [Data Dictionary Template](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/docs/04_data_dictionary_template.md)
- [Methodology Notes](/mnt/c/Users/javed/Documents/Projects/Analytical Finance and Machine Learning/docs/05_methodology_notes.md)
