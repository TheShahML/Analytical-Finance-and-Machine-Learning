# Methodology Notes

## Purpose

This memo records methodological defaults and guardrails for the project. It is meant to keep the workflow disciplined while the team is still in the data and design stage.

## General Principles

- keep sentiment and communication style conceptually separate
- prefer interpretable baselines before complex modeling
- document cleaning and filtering decisions explicitly
- treat exploratory analysis as exploration, not proof
- move reusable logic into `src/` rather than leaving it buried in notebooks

## Data Pipeline Stages

### Stage 1: Raw Data Intake

- identify the canonical raw transcript source
- record provenance and file location
- capture the observed schema

### Stage 2: Validation and Audit

- check required fields
- review duplicates and missingness
- inspect identifier quality and date fields
- log any filtering candidates

### Stage 3: Cleaned Transcript Dataset

- define the transcript-level analysis unit
- standardize key identifiers and dates
- preserve raw-to-clean mapping where possible

### Stage 4: Exploratory NLP and EDA

- descriptive counts and coverage summaries
- keyword and theme exploration
- preliminary topic discovery
- early sentiment and style prototypes

### Stage 5: Finance-Ready Merge Construction

- define event dates
- merge transcript observations to returns and controls
- document sample restrictions for each empirical table

## Transcript Cleaning Decisions

Decisions that should be recorded explicitly:

- how missing transcript text is handled
- whether duplicate or corrected transcripts are removed or flagged
- how dates are parsed and prioritized
- whether prepared remarks and Q&A are separated
- whether legal boilerplate is kept, removed, or flagged
- how transcript length and word counts are recomputed or validated

The project should avoid silent cleaning choices.

## Basic NLP Exploration Options

Suitable early methods:

- keyword counts
- seed dictionaries
- phrase-pattern inspection
- transcript length-normalized counts
- basic clustering or topic exploration

These methods are useful because they are easy to inspect and can help clarify which constructs are worth formalizing.

## Topic Modeling Notes

Topic modeling should be used cautiously and mainly for organization and exploratory insight at first.

Practical uses:

- identify broad recurring themes across calls
- help separate business-content topics from communication-style measures
- surface candidate subdomains for later analysis

Cautions:

- topic mixtures can reflect industry or time-period composition
- topic models are not direct measures of transparency, directness, or evasiveness
- topic outputs should not be overinterpreted in the early stage

## Sentiment Baseline Notes

Sentiment is a benchmark, not the full transcript representation.

Guidelines:

- keep positive, negative, and neutral components if available
- avoid reducing everything to a single net score too early
- compare sentiment outputs against style measures rather than combining them immediately
- document the exact model or dictionary used

## Communication-Style Scoring Notes

Candidate style families include:

- hedging or uncertainty
- vagueness
- directness or specificity
- transparency
- attribution or accountability
- evasiveness or deflection

Recommended early sequence:

1. build transparent dictionary and phrase-based prototypes
2. normalize by transcript or section length
3. inspect high-scoring examples manually
4. move to model-assisted methods only after baseline measures are understandable

## Event Study Notes

Important design choices to document:

- event date definition
- return source
- estimation window
- event window
- abnormal-return construction
- treatment of after-hours or timing ambiguity if relevant

The event-study pipeline should be explicit about how a transcript observation becomes an event observation.

## Panel Regression Notes

Important design choices to document:

- unit of analysis
- dependent-variable horizon
- feature timing
- control set
- fixed effects or clustering choices
- sample restrictions and outlier handling

The first panel specifications should stay simple enough to diagnose.

## Reproducibility and Versioning

Minimum expectations:

- keep reusable logic in version-controlled modules
- document major feature-definition changes
- store generated tables and figures in consistent output locations
- record the exact raw input used for each cleaned dataset
- avoid manual, undocumented notebook-only transformations when possible

## What Should Be Notebook-Based vs Module-Based

Notebook-based work:

- exploratory plots
- manual inspection
- descriptive summaries
- early hypothesis generation

Module-based work:

- data loading
- schema checks
- validation summaries
- reusable feature construction
- finance merge logic
- event-study and regression interfaces

If code is likely to be reused or tested, it should move into `src/`.
