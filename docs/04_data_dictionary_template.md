# Data Dictionary Template

## Purpose

This template is for documenting the transcript dataset and any derived tables used later in feature construction or finance tests. It should be updated during the data audit, not filled with guessed values.

## Dataset Metadata

| Field | Value |
| --- | --- |
| Dataset name |  |
| Current raw file path |  |
| Canonical raw file path |  |
| Source or provenance note |  |
| Observation unit |  |
| Time coverage |  |
| Last validation date |  |
| Primary owner or maintainer |  |

## Raw Transcript Fields

Use this table for fields that appear in the raw transcript source.

| Field name | Type | Description | Required for audit | Required for downstream analysis | Missingness notes | Validation notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

## Cleaned Transcript Fields

Use this table for the cleaned transcript-level dataset that will be used in EDA and later merges.

| Field name | Type | Description | Source field or rule | Included in cleaned dataset | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

## Identifier Fields

Document every identifier that may be used for transcript, firm, or finance linkage.

| Identifier | Level | Example format | Intended use | Expected uniqueness | Missingness notes | Comments |
| --- | --- | --- | --- | --- | --- | --- |
| `transcriptid` |  |  |  |  |  |  |
| `companyid` |  |  |  |  |  |  |
| `ticker` |  |  |  |  |  |  |
| `permno` |  |  |  |  |  |  |
| `gvkey` |  |  |  |  |  |  |
| `ibes_ticker` |  |  |  |  |  |  |

## Transcript Metadata Fields

Document metadata fields that describe the transcript but are not themselves derived NLP features.

| Field name | Description | Notes on interpretation | Notes on quality or ambiguity |
| --- | --- | --- | --- |
| `headline` |  |  |  |
| `event_type` |  |  |  |
| `call_date` |  |  |  |
| `actual_call_date` |  |  |  |
| `transcriptcreationdate_utc` |  |  |  |
| `mostimportantdateutc` |  |  |  |
| `transcript_length` |  |  |  |
| `word_count` |  |  |  |

## Derived NLP Features

Use one row per feature or feature family.

| Feature name | Feature family | Unit of analysis | Construction summary | Normalization | Planned module | Notes |
| --- | --- | --- | --- | --- | --- | --- |
|  | sentiment |  |  |  |  |  |
|  | hedging or uncertainty |  |  |  |  |  |
|  | vagueness |  |  |  |  |  |
|  | directness or specificity |  |  |  |  |  |
|  | transparency |  |  |  |  |  |
|  | attribution |  |  |  |  |  |
|  | evasiveness or deflection |  |  |  |  |  |

## Finance and Event Variables

Use this section for market, event, and control variables needed later for empirical tests.

| Variable name | Role | Intended use | Source | Required for event study | Required for panel tests | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `event_date` |  |  |  |  |  |  |
| `ret` |  |  |  |  |  |  |
| `abnormal_return` |  |  |  |  |  |  |
| `car_window` |  |  |  |  |  |  |
| `earnings_surprise` |  |  |  |  |  |  |
| `size` |  |  |  |  |  |  |
| `momentum` |  |  |  |  |  |  |
| `volatility` |  |  |  |  |  |  |

## Missingness and Filtering Notes

Document the main data-quality issues and the rules used to retain or exclude observations.

| Issue | Affected fields | Planned handling | Final decision | Notes |
| --- | --- | --- | --- | --- |
| Missing transcript text |  |  |  |  |
| Missing event date |  |  |  |  |
| Duplicate transcript IDs |  |  |  |  |
| Missing finance identifiers |  |  |  |  |
| Non-earnings event types |  |  |  |  |

## Derived Dataset Inventory

Track each analysis table created from the raw transcript source.

| Dataset name | Intended location | Observation unit | Upstream inputs | Main purpose | Owner | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Raw transcript master | `data/raw/` |  |  |  |  |  |
| Cleaned transcript master | `data/processed/` |  |  |  |  |  |
| Transcript feature table | `data/processed/` |  |  |  |  |  |
| Event-study input table | `data/interim/` or `data/processed/` |  |  |  |  |  |
| Panel analysis sample | `data/processed/` |  |  |  |  |  |
