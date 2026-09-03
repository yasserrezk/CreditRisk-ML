# CreditRisk-ML

CreditRisk-ML prepares and analyzes credit-risk data from the "Give Me Some
Credit" dataset. The project focuses on safe, reproducible preprocessing for
downstream modeling: cleaning invalid entries, engineering debt-related
features, and imputing missing values using models fit only on training data.

## Quickstart

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the end-to-end preprocessing pipeline from the repository root:

```bash
python src/data/data_cleaning/clean_data.py
```

Outputs are written to `data/processed/processed_train.csv` and
`data/processed/processed_test.csv`.

## What this repo contains

- Raw and processed CSV datasets under `data/`.
- Reusable data loading utilities in `src/data/data_fetching`.
- Deterministic cleaning, feature-engineering, and imputation code in
  `src/data/data_cleaning`, `src/features`, and `src/data/data_cleaning`.
- Notebooks for exploration in `notebooks/` and `src/data/data_understanding`.
- A Power BI dashboard snapshot in `dashboard/` and example artifacts under
  `artifacts/` (serialized models and exports).

## Repository layout (high level)

CreditRisk-ML/
- data/
  - raw/              # Source CSVs (do not modify)
  - processed/        # Generated outputs from the pipeline
- src/
  - data/
    - data_cleaning/      # `clean_data.py`, pipeline entrypoint
    - data_fetching/      # CSV loaders
    - data_understanding/ # analysis notebooks
  - features/             # `feature_engineering.py`, `smart_imputation.py`
  - models/               # model training & evaluation (planned)
- notebooks/             # EDA and analysis notebooks
- dashboard/             # Power BI file + screenshots
- configs/               # Configuration templates (planned)
- tests/                 # Tests (planned)
- artifacts/             # Saved models and evaluation bundles

## Data pipeline (summary)

The preprocessing script implements these steps:

1. Load raw CSVs from `data/raw/`.
2. Normalize ID column and remove exact duplicate training rows.
3. Apply business-rule validation and clip implausible test values (age,
   utilization, delinquency counts).
4. Derive `MonthlyDebt` from `DebtRatio` and `MonthlyIncome`, then remove
   `DebtRatio`.
5. Fit an `IterativeImputer` (with `RandomForestRegressor`) on training data
   and reuse the fitted imputer for test rows to avoid data leakage.
6. Derive `NonRealEstateLoans` and enforce non-negative constraints.
7. Save processed outputs to `data/processed/`.

Run the pipeline from the repository root (the script resolves the root by
searching upward for the `src` directory).

## Important files and modules

- `src/data/data_cleaning/clean_data.py` — pipeline orchestration and I/O.
- `src/data/data_cleaning/data_cleaning_pipeline.py` — normalization and
  validation rules.
- `src/features/feature_engineering.py` — `MonthlyDebt` and derived fields.
- `src/features/smart_imputation.py` — iterative random-forest imputation.
- `src/data/data_fetching/load_data.py` — CSV loaders for raw/processed.

## Notes for contributors

- Do not edit files in `data/raw/`; they are the canonical source data.
- Refit the imputer only on training data; reuse it for test rows to avoid
  leakage.
- Regenerate `data/processed/` whenever cleaning or feature logic changes.
- Add unit tests under `tests/` when new processing behavior is introduced.

## Dependencies

Create a virtual environment and install the common dependencies listed in
`requirements.txt` (pandas, numpy, scikit-learn, jupyter). If `requirements.txt`
is missing, I can generate one from the codebase.

## Next steps I can help with

- Add `requirements.txt` and a lightweight `Makefile` or CLI wrapper.
- Add a `tests/` smoke test that runs the pipeline and verifies outputs.
- Scaffold a model training script and evaluation workflow.

If you want one of these next steps, tell me which and I'll implement it.
