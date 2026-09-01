import pandas as pd
import numpy as np

# Enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor


def create_imputer():

    rf_estimator = RandomForestRegressor(
        n_estimators=20,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    imputer = IterativeImputer(
        estimator=rf_estimator,
        max_iter=3,
        random_state=42
    )

    return imputer


def impute_basic_features(df, imputer, fit=False):

    smart_cols = [
        'age',
        'NumberRealEstateLoansOrLines',
        'NumberOfOpenCreditLinesAndLoans',
        'MonthlyIncome',
        'MonthlyDebt',
        'NumberOfDependents'
    ]

    if fit:
        # Fit ONLY on training data
        df[smart_cols] = imputer.fit_transform(df[smart_cols])
    else:
        # Use the already fitted imputer
        df[smart_cols] = imputer.transform(df[smart_cols])

    # Number of dependents must be a non-negative integer
    df['NumberOfDependents'] = (
        df['NumberOfDependents']
        .round()
        .clip(lower=0)
    )

    # Monthly income cannot be negative
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(lower=0)

    # Monthly debt cannot be negative
    df['MonthlyDebt'] = df['MonthlyDebt'].clip(lower=0)

    return df


def run_imputation_pipeline(df, imputer, fit=False):

    print("\nStarting Smart Imputation Pipeline...")

    df = impute_basic_features(
        df,
        imputer=imputer,
        fit=fit
    )

    print("Smart Imputation Complete.\n")

    return df