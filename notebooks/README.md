# Notebook Guide

## Main Workflow

Use these notebooks in order for the active buyback project:

1. `00_transcript_and_revenue_data_pull.ipynb`
   WIP provenance pull notebook. Only needed if you are rebuilding upstream data.
2. `01_data_audit_and_validation.ipynb`
   Checks the cleaned transcript base and writes validated outputs.
3. `02_initial_eda.ipynb`
   General exploratory work on the cleaned transcript panel.
4. `03_sentiment_baseline.ipynb`
   Baseline sentiment work for full transcripts and buyback text.
5. `04_clarity_signal_prototyping.ipynb`
   Q&A clarity feature prototyping.
6. `05_event_study_baseline.ipynb`
   Event-study baseline with the current project window design.
7. `06_buyback_sentiment_clarity.ipynb`
   Master notebook for the current research pipeline.

## Archive

These notebooks are not part of the main workflow now:

- `90_keyword_and_theme_exploration.ipynb`
- `91_topic_modeling.ipynb`
- `98_panel_tests_legacy.ipynb`

`98_panel_tests_legacy.ipynb` is archived because panel regression is no longer in the active design.
