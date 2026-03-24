# Task 01 Steering: Shared Dataset Audit, Validation, and Cleaning

## Purpose

This document steers the first project workstream: build a trustworthy shared transcript dataset that the team can all use for later EDA, feature design, and finance tests.

Task 01 is no longer just a schema check. It now includes:

- data audit
- data validation
- targeted cleaning
- post-cleaning descriptive summaries
- stable saved outputs for downstream notebooks

The guiding principle is simple: before we analyze transcript signals, we need to know what the dataset contains, what is wrong with it, what was fixed, and what the cleaned shared file represents.

## Professor Objective in Plain Language

For the class, the immediate requirement is not a finished research result. The requirement is a clean and well-understood transcript corpus.

That means Task 01 should produce:

- a repeatable notebook that loads the transcript file and documents the key issues
- a reusable `src/` implementation for the main audit and cleaning rules
- a cleaned shared dataset with transparent filtering rules
- saved tables and figures that summarize what was kept, what was dropped, and what still needs attention

## Task 01 Notebook Scope

Primary notebook:

- `notebooks/01_data_audit_and_validation.ipynb`

Primary reusable modules:

- `src/data/load_transcripts.py`
- `src/data/validate_transcripts.py`
- `src/data/clean_transcripts.py`
- `src/data/build_panel.py`
- `src/utils/paths.py`

Task 01 should visibly do four things in order:

1. Load the raw transcript dataset and confirm the working schema.
2. Prove or document the main data issues that affect the shared sample.
3. Apply the agreed cleaning rules.
4. Validate and summarize the cleaned output.

## Agreed Task 01 Cleaning Rules

These are the current working rules for the shared cleaned dataset.

### Rule 1: Restrict the sample window by `call_date`

Working rule:

- keep only rows with `call_date >= 2010-01-01`

Why:

- the intended shared sample now starts in 2010
- older calls outside that window should not remain in the cleaned shared file

### Rule 2: Deduplicate likely transcript revisions

Working rule:

- within each `(ticker, call_date)` group, keep the longest transcript

Why:

- duplicate rows appear to be multiple versions of the same event
- the longest transcript is the current practical proxy for the most complete call record

Important note:

- this is an operational cleaning rule, not a proven universal truth
- if later validation shows a better event-level deduplication rule, the pipeline can be updated

### Rule 3: Remove exact close-to-open price matches

Working rule:

- drop rows where `close_price_call_day == open_price_next_day`

Why:

- these rows are being treated as likely stale or fallback price observations in the current shared sample
- this rule matches the current class-cleaning workflow used by the team

Important note:

- this is a sample-construction choice for the cleaned shared dataset
- it should be documented clearly so later finance testing is not treated as if it started from a raw, untouched file

## Required Task 01 Deliverables

### 1. Raw Audit

Deliverable:

- raw file load summary
- observed schema summary
- date range and coverage summary
- basic identifier availability summary

Status:

| Item | Status | Notes |
| --- | --- | --- |
| Raw loading notebook structure | complete | Implemented |
| Reusable loader module | complete | Implemented |
| Raw schema inspection | in progress | Depends on latest full run outputs |
| Canonical raw-source decision | not started | Current raw source still needs formal confirmation |

### 2. Issue Proof and Validation

Deliverable:

- duplicate-event diagnostic
- pre-period leakage diagnostic
- exact-zero-return diagnostic
- saved tables supporting each issue review

Status:

| Item | Status | Notes |
| --- | --- | --- |
| Duplicate-event diagnostic | complete | Implemented in notebook and `src/data/clean_transcripts.py` |
| Pre-period leakage diagnostic | complete | Implemented in notebook and `src/data/clean_transcripts.py` |
| Exact-zero-return diagnostic | complete | Implemented in notebook and `src/data/clean_transcripts.py` |
| Full-run issue review signoff | in progress | Needs team review on latest outputs |

### 3. Cleaned Shared Dataset

Deliverable:

- cleaned transcript dataset written to stable project paths
- cleaning log with row counts before and after each rule
- transcript-level usability flags for downstream notebooks

Status:

| Item | Status | Notes |
| --- | --- | --- |
| Cleaning pipeline code | complete | Implemented |
| Cleaning log output | complete | Implemented |
| Cleaned CSV output | complete | Implemented |
| Final canonical cleaned dataset signoff | in progress | Needs team confirmation |

### 4. Post-Cleaning EDA

Deliverable:

- counts by year
- firm concentration summary
- transcript-length summary
- simple keyword/theme summary

Status:

| Item | Status | Notes |
| --- | --- | --- |
| Counts by year | complete | Implemented in notebook output flow |
| Firm summary | complete | Implemented in notebook output flow |
| Transcript-length summary | complete | Implemented in notebook output flow |
| Keyword exploration | complete | Implemented in notebook output flow |
| Topic exploration | not started | Deferred to later notebook |

## Validation Checklist for the Cleaned Output

Use this checklist when reviewing a Task 01 run.

| Validation item | Status | Notes |
| --- | --- | --- |
| Raw file loaded successfully | in progress | Confirm on latest full run |
| Schema observed and saved | complete | Implemented |
| Duplicate-event issue summarized | complete | Implemented |
| Pre-period issue summarized | complete | Implemented |
| Exact-zero-return issue summarized | complete | Implemented |
| Cleaning steps logged with row counts | complete | Implemented |
| No duplicate `(ticker, call_date)` rows remain | in progress | Validate on latest saved cleaned output |
| Earliest `call_date` is on or after 2010-01-01 | in progress | Validate on latest saved cleaned output |
| No exact close/open matches remain | in progress | Validate on latest saved cleaned output |
| Cleaned dataset saved for reuse | complete | Implemented |
| Audit tables saved for reuse | complete | Implemented |
| Figures saved for reuse | complete | Implemented |

## Expected Task 01 Outputs

The notebook should show key results inline and also write them to disk.

Saved datasets:

- cleaned transcript parquet
- cleaned transcript CSV
- flagged cleaned transcript parquet
- usable transcript parquet
- lightweight transcript event panel parquet

Saved tables:

- cleaning log
- validation summary
- missingness summary
- identifier match summary
- duplicate issue tables
- pre-period issue tables
- zero-return issue tables
- post-cleaning EDA tables

Saved figures:

- counts by year
- top firms
- transcript-length distribution

## Open Questions That Still Need Team Decisions

- Is `2010-01-01` the final intended shared sample start date, or should the team revise it again later?
- Is the longest-transcript rule always the right way to resolve event-level revisions?
- Should the exact-zero-return filter remain part of the shared cleaned dataset, or move to a finance-ready subset later?
- Which date field should be treated as the final event date for downstream finance tests?
- Can prepared remarks and Q&A be separated later from the cleaned transcript text?

## Task 01 Exit Criteria

Task 01 is in good shape when all of the following are true:

- the notebook runs end to end on the intended dataset
- the three issue checks produce interpretable evidence
- the cleaning log clearly shows what changed
- the cleaned dataset can be reused by later notebooks without redoing raw-file logic
- the team agrees on what the cleaned file represents and what it does not represent
