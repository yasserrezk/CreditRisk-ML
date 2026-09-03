# Data processing

This package contains the data-ingestion, cleaning, and data-understanding
parts of the project.

## Subdirectories

- `data_fetching/`: loads raw and processed CSV files with pandas.
- `data_cleaning/`: validates records, handles duplicates, and orchestrates
  preparation of train and test data.
- `data_understanding/`: Jupyter notebooks for inspecting dataset quality and
  processed outputs.

The main end-to-end entry point is
`data_cleaning/data_cleaning_pipeline.py`, which writes outputs to
`data/processed/`. Raw source files under `data/raw/` should remain unchanged.
