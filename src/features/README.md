# Feature engineering

This package transforms cleaned borrower data into model-ready features.

## Modules

- `feature_engineering.py`
  - Creates `MonthlyDebt` from `DebtRatio` and `MonthlyIncome`.
  - Removes the redundant raw `DebtRatio` field.
  - Creates `NonRealEstateLoans` from total open credit lines and real-estate
    loans or lines.
- `smart_imputation.py`
  - Uses scikit-learn's `IterativeImputer`.
  - Uses a `RandomForestRegressor` estimator to estimate missing numeric values.
  - Fits only on training data and transforms test data with the fitted imputer.
  - Enforces non-negative income, debt, and dependent-count constraints.

The feature order matters: debt features must be created before imputation,
because `MonthlyDebt` is one of the imputed fields.