# CreditRisk-ML

Machine learning pipeline for credit-risk scoring on the **"Give Me Some
Credit"** dataset. Given a borrower's financial history, the model estimates
the probability that they will experience serious delinquency (90+ days
past due) within the next two years — the `SeriousDlqin2yrs` target.

The project covers the full workflow end to end:

```
raw data → cleaning → feature engineering → imputation → model training
         → evaluation → artifacts → Streamlit dashboard / Power BI report
```

## Results

Three classifiers are trained and compared on a held-out split of the
training data (metrics from `artifacts/model_comparison.csv`):

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|----------------------|---------:|----------:|-------:|-------:|--------:|
| **XGBoost**          | 0.943    | 0.588     | 0.128  | 0.210  | **0.863** |
| Logistic Regression  | 0.796    | 0.189     | 0.745  | 0.302  | 0.854   |
| SVM                  | 0.937    | 0.071     | 0.005  | 0.010  | 0.744   |

XGBoost has the best ROC-AUC and is selected as the best model
(`artifacts/best_model.pkl`), but note the trade-off: it recalls only ~13%
of true defaulters at the default 0.5 threshold, while Logistic Regression
recalls ~75% at the cost of many more false positives. The target class is
rare (~6.7% positive rate), so threshold choice matters — the dashboard lets
you explore this trade-off interactively.

## Repository layout

```
CreditRisk-ML/
├── data/
│   ├── raw/                 # Original CSVs — immutable inputs
│   └── processed/           # Cleaned/imputed/feature-engineered outputs
├── src/
│   ├── data/
│   │   ├── data_fetching/       # CSV loaders (raw + processed)
│   │   ├── data_cleaning/       # Validation rules + pipeline entrypoint
│   │   └── data_understanding/  # Data-quality notebooks
│   ├── features/             # Feature engineering + imputation
│   ├── models/                # Training, tuning, evaluation, config
│   └── api/                   # Reserved for a future prediction API
├── notebooks/                # EDA and model-training notebooks
├── dashboard/                 # Streamlit app + Power BI report
├── configs/                   # Reserved for config files (hyperparameters, thresholds)
├── artifacts/                 # Saved models, eval bundle, predictions
├── tests/                     # Reserved for pytest suite
└── requirements.txt
```

## Quickstart

**1. Set up the environment**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the data pipeline** (cleans, engineers features, imputes)

```bash
python src/data/data_cleaning/clean_data.py
```

Produces `data/processed/processed_train.csv` and
`data/processed/processed_test.csv`.

**3. Train the models**

```bash
python -m src.models.train
```

Trains Logistic Regression, XGBoost, and SVM (tuning via
`RandomizedSearchCV`), evaluates each, saves the best model, and scores the
real test set. Outputs go to `artifacts/`. If a model's `.joblib` file
already exists in `artifacts/models/`, training is skipped and the saved
model is loaded instead — delete the file to force retraining.

**4. Explore results in the dashboard**

```bash
streamlit run dashboard/app.py
```

The dashboard reads the artifacts produced in step 3 — it does not retrain
anything, so it loads instantly. It shows model comparison, ROC/PR curves,
a confusion matrix with an adjustable classification threshold, feature
importances, and the scored test-set predictions (downloadable as CSV).

## Data

Sourced from the ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit)
Kaggle competition.

| File | Rows | Description |
|---|---:|---|
| `data/raw/training.csv` | 150,000 | Labeled borrower records (has `SeriousDlqin2yrs`) |
| `data/raw/test.csv` | 101,503 | Unlabeled borrower records for scoring |
| `data/raw/testsampleweight.csv` | — | Sample weights for the test rows |

The target is heavily imbalanced: **~6.7%** of training borrowers are
positive (`SeriousDlqin2yrs = 1`). This is handled with `class_weight`
(Logistic Regression, SVM) and `scale_pos_weight` (XGBoost), and models are
scored on ROC-AUC rather than accuracy alone.

Raw features include revolving credit utilization, age, count of
30–59/60–89/90+ day delinquencies, debt ratio, monthly income, number of
open credit lines/loans, number of real-estate loans, and number of
dependents.

## Pipeline details

### 1. Cleaning (`src/data/data_cleaning/`)

- Renames the unnamed index column to `Id`.
- Drops exact duplicate rows (ignoring `Id`), training data only.
- **Training data**: drops implausible rows — `age < 21`,
  `RevolvingUtilizationOfUnsecuredLines > 1`, or any of the three
  delinquency counts `> 20`.
- **Test data**: instead of dropping rows (every row must be scored),
  clips the same values to valid ranges.
- The same checks are re-applied *after* imputation, since imputed values
  can otherwise violate these business rules.

### 2. Feature engineering (`src/features/feature_engineering.py`)

- **`MonthlyDebt`** replaces `DebtRatio`: if `MonthlyIncome` is missing or
  zero, `DebtRatio` is treated as an absolute debt figure; otherwise
  `MonthlyDebt = DebtRatio × MonthlyIncome`. `DebtRatio` is then dropped —
  it produced unstable values after imputation and is redundant with
  `MonthlyIncome` + `MonthlyDebt`.
- **`NonRealEstateLoans`** = `NumberOfOpenCreditLinesAndLoans −
  NumberRealEstateLoansOrLines`, clipped at zero, as a proxy for reliance
  on unsecured debt.

### 3. Imputation (`src/features/smart_imputation.py`)

- Missing values in `age`, `NumberRealEstateLoansOrLines`,
  `NumberOfOpenCreditLinesAndLoans`, `MonthlyIncome`, `MonthlyDebt`, and
  `NumberOfDependents` are filled with scikit-learn's `IterativeImputer`
  (`RandomForestRegressor` estimator, 20 trees, max depth 8).
- The imputer is **fit only on training data** and reused (transform-only)
  on the test set, to avoid leakage.
- Post-imputation constraints: `NumberOfDependents` rounded to a
  non-negative integer; `MonthlyIncome` and `MonthlyDebt` clipped at zero.

Processed output: 13 columns, 145,698 training rows (after dedup/filtering)
and 101,503 test rows.

### 4. Modeling (`src/models/`)

- `config.py` — paths, target column (`SeriousDlqin2yrs`), random seed (42).
- `data_loader.py` — loads processed CSVs, does a stratified 80/20
  train/test split.
- `model/` — one module per algorithm (`logistic_regression.py`,
  `xgboost_model.py`, `svm_model.py`), each exposing a baseline builder, a
  tuning search-space, and a search estimator.
- `tuning.py` — shared `RandomizedSearchCV` wrapper (5-fold stratified CV,
  scored on ROC-AUC).
- `evaluation.py` — accuracy, precision, recall, F1, ROC-AUC, plus a
  confusion-matrix plot.
- `train.py` — orchestrates the above: trains/tunes all three models
  (SVM is trained on a subsample of 80,000 rows for tractability), compares
  them, saves the best one, and scores the real test set into
  `artifacts/test_predictions.csv`.

## Artifacts

Generated by `python -m src.models.train`:

| File | Contents |
|---|---|
| `best_model.pkl` / `best_model.joblib` | Best model by ROC-AUC (currently XGBoost) |
| `models/*.joblib` | Every trained model, cached individually |
| `model_comparison.csv` | Metrics table for all models |
| `eval_bundle.joblib` | Test-set probabilities, feature importances, and comparison data used by the dashboard |
| `test_predictions.csv` | Predicted probability/class for every row in `data/raw/test.csv` |

## Dashboard

- **Streamlit** (`dashboard/app.py`) — interactive model comparison, ROC/PR
  curves, adjustable-threshold confusion matrix, feature importance, and
  downloadable test-set predictions. Reads only from `artifacts/`.
- **Power BI** (`dashboard/powerbi/Credit_Risk_Analysis.pbix`) — a static
  report; open it in Power BI Desktop. Screenshots are included as a
  preview for anyone without Power BI installed.

## Project status / roadmap

- `src/api/` — not yet implemented. Intended to expose the model via
  FastAPI for real-time scoring.
- `configs/` — not yet populated. Intended for externalized hyperparameters
  and thresholds currently hardcoded in `src/models/config.py`.
- `tests/` — not yet populated. Intended pytest coverage for the cleaning
  rules, feature engineering, and imputation logic.

## Development notes

- Never edit files under `data/raw/` — treat them as immutable inputs.
- Refit the imputer on training data only; always reuse the fitted imputer
  (never refit) on test data.
- Regenerate `data/processed/` after any change to cleaning, imputation, or
  feature-engineering logic, and rerun training afterward so artifacts stay
  consistent with the data.
- Delete the relevant file in `artifacts/models/` to force a model to
  retrain instead of loading the cached version.

## Dependencies

See `requirements.txt`: pandas, numpy, scikit-learn, xgboost, joblib,
matplotlib, seaborn, jupyter. The dashboard additionally requires
`streamlit` and `plotly` (not currently pinned in `requirements.txt` —
install with `pip install streamlit plotly` if you hit an import error).
