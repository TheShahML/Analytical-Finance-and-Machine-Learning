# Task 00 Notebook Note

The provenance notebook for the transcript pull and revenue enrichment step now lives at:

- `notebooks/00_transcript_and_revenue_data_pull.ipynb`

That notebook is the project reference for:

1. pulling the raw transcript corpus from WRDS / CIQ
2. enriching the raw corpus with identifiers, prices, and revenue-related fields
3. writing the raw project dataset to `data/raw/`

Current intended raw output names:

- `data/raw/earnings_calls_full_2010_onward.csv`
- `data/raw/earnings_calls_full_2010_onward_with_revenue.csv`
