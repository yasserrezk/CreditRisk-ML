# Raw data

This directory contains the original source CSV files:

- `training.csv`: labeled borrower records, including `SeriousDlqin2yrs`.
- `test.csv`: borrower records without target labels.
- `testsampleweight.csv`: sample weights associated with test rows.

These files are the inputs to
`src/data/data_cleaning/clean_data.py`. Treat them as immutable: do not edit,
overwrite, or manually clean them. Store all generated data in
`data/processed/` instead.