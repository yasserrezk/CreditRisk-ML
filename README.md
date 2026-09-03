# CreditRisk-ML

CreditRisk-ML is a credit-risk data preparation and exploratory analysis project
based on the Give Me Some Credit dataset. It prepares raw borrower records for
downstream modeling by cleaning invalid values, engineering debt-related
features, and imputing missing values without fitting on the test set.

## Project status

The repository currently contains:

- Raw and processed training/test CSV files.
- A reusable data-loading module.
- Cleaning, feature-engineering, and smart-imputation pipelines.
- Jupyter notebooks for data understanding and exploratory analysis.
- A Power BI dashboard and exported screenshots.

The `src/models`, `src/api`, `configs`, `tests`, and `artifacts` directories are
documented extension points. Their README files describe the intended contents,
but implementation files are not currently included.

## Repository structure

```text
CreditRisk-ML/
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── README.txt
│   │   ├── training.csv              # Training records and target column
│   │   ├── test.csv                  # Test records without target values
│   │   └── testsampleweight.csv      # Test-row sample weights
│   └── processed/
│       ├── README.txt
│       ├── processed_train.csv       # Cleaned and engineered training data
│       └── processed_test.csv        # Cleaned and engineered test data
├── src/
│   ├── api/
│   │   └── README.txt                # Planned FastAPI serving layer
│   ├── data/
│   │   ├── README.txt
│   │   ├── data_cleaning/
│   │   │   ├── clean_data.py         # Dataset-level processing entry point
│   │   │   └── data_cleaning_pipeline.py
│   │   ├── data_fetching/
│   │   │   └── load_data.py          # Raw and processed CSV loaders
│   │   └── data_understanding/
│   │       ├── data_understanding.ipynb
│   │       └── data_processed_understanding.ipynb
│   ├── features/
│   │   ├── README.txt
│   │   ├── feature_engineering.py    # MonthlyDebt and NonRealEstateLoans
│   │   └── smart_imputation.py        # Iterative random-forest imputation
│   └── models/
│       └── README.txt                # Planned training and evaluation layer
├── notebooks/
│   ├── README.txt
│   └── EDA.ipynb                     # Exploratory data analysis
├── dashboard/
│   ├── README.txt
│   ├── Credit_Risk_Analysis.pbix
│   └── Screenshot *.png
├── configs/
│   └── README.txt                    # Planned YAML/JSON configuration files
├── tests/
│   └── README.txt                    # Planned pytest test suite
└── artifacts/
    └── README.txt                    # Planned serialized models and preprocessors
```

Generated Python cache files such as `__pycache__/` and `*.pyc` are excluded by
`.gitignore`.

## Data pipeline

Run the complete preparation workflow from the repository root:

```bash
python src/data/data_cleaning/clean_data.py
```

The workflow:

1. Loads `data/raw/training.csv` and `data/raw/test.csv`.
2. Renames `Unnamed: 0` to `Id` when necessary.
3. Removes duplicate training rows (excluding the ID).
4. Removes invalid training values and clips corresponding test values:
   - age below 21;
   - revolving utilization above 1;
   - delinquency counts above 20.
5. Creates `MonthlyDebt` from `DebtRatio` and `MonthlyIncome`, then removes
   `DebtRatio`.
6. Fits one `IterativeImputer` with a `RandomForestRegressor` on training data.
7. Uses that same fitted imputer for test data to prevent data leakage.
8. Creates `NonRealEstateLoans` and clips negative results to zero.
9. Writes `data/processed/processed_train.csv` and
   `data/processed/processed_test.csv`.

The script locates the project root by searching upward for the `src`
directory, so it should be run from the repository or a child directory.

## Data columns

The raw datasets use the following fields:

| Column | Description |
| --- | --- |
| `Id` | Borrower identifier |
| `SeriousDlqin2yrs` | Target: serious delinquency within two years (training data only) |
| `RevolvingUtilizationOfUnsecuredLines` | Unsecured revolving credit utilization |
| `age` | Borrower age |
| `NumberOfTime30-59DaysPastDueNotWorse` | 30–59 day delinquency count |
| `DebtRatio` | Raw debt-to-income ratio, replaced during processing |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines and loans |
| `NumberOfTimes90DaysLate` | 90+ day delinquency count |
| `NumberRealEstateLoansOrLines` | Real-estate-backed loans or lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | 60–89 day delinquency count |
| `NumberOfDependents` | Number of dependents |

Processed files replace `DebtRatio` with `MonthlyDebt` and add
`NonRealEstateLoans`. The processed test file retains an empty
`SeriousDlqin2yrs` column because the source test data has no labels.

## Python modules

- `load_data.py`: loads raw or processed CSV files into pandas DataFrames.
- `data_cleaning_pipeline.py`: normalizes IDs, handles duplicates, and applies
  business-rule validation.
- `feature_engineering.py`: derives absolute monthly debt and non-real-estate
  credit lines.
- `smart_imputation.py`: imputes selected numeric fields with an iterative
  random-forest estimator and enforces non-negative business constraints.
- `clean_data.py`: orchestrates the complete train/test workflow and saves
  outputs.

## Analysis and dashboard

- Use [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) for exploratory analysis.
- Use the notebooks under
  [`src/data/data_understanding/`](src/data/data_understanding/) to inspect raw
  and processed data.
- Open `dashboard/Credit_Risk_Analysis.pbix` with Power BI Desktop to explore
  the included dashboard. The PNG files in the same directory are snapshots.

## Development notes

- Do not modify files in `data/raw/`; they are the source datasets.
- Regenerate processed files after changing cleaning or feature-engineering
  logic.
- Keep the imputer fit on training data only and reuse it for test data.
- Install the project dependencies in your Python environment before running
  the pipeline: pandas, NumPy, scikit-learn, and Jupyter for notebooks.
