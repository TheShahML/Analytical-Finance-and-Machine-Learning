# Notebook Guide

## Main workflow

Run in numeric order:

1. `00_transcript_and_revenue_data_pull.ipynb` — WRDS/raw provenance (optional if `FINAL.csv` exists).
2. `01_data_audit_and_validation.ipynb` — Audit the transcript base; writes validated outputs.
3. `02_initial_eda.ipynb` — Exploratory work on the audited panel (keywords, topics).
4. `03_buyback_sentiment_clarity.ipynb` — **Master buyback pipeline** on `FINAL.csv` (sentiment, clarity, event study).
5. `04_buyback_announcement_classification_ollama.ipynb` — LLM classification (Ollama); run before classification EDA.
6. `05_sentiment_clarity_eda_outputs.ipynb` — Sentiment × clarity EDA and figures (e.g. `outputs/eda/`).
7. `06_classification_eda_outputs.ipynb` — Scaffold for EDA on outputs from notebook04.

## Archive

Not part of the active buyback line:

- `90_keyword_and_theme_exploration.ipynb`
- `91_topic_modeling.ipynb`
- `98_panel_tests_legacy.ipynb` — legacy panel design; panel regression module removed from `src/`.
