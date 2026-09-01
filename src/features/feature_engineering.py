import pandas as pd
import numpy as np

def engineer_debt_features(df):
    """
    Creates a 'MonthlyDebt' feature from DebtRatio and MonthlyIncome,
    then drops 'DebtRatio' entirely (MonthlyDebt replaces it as the
    debt-related feature going forward).
    """
    print("--- Engineering Debt Features ---")

    # 1. Identify rows where income is missing or exactly zero
    no_income_mask = df['MonthlyIncome'].isna() | (df['MonthlyIncome'] == 0)

    # 2. Extract Absolute Debt BEFORE imputing income
    # If no income, DebtRatio is actually the absolute debt.
    # Otherwise, it's (DebtRatio * MonthlyIncome)
    df['MonthlyDebt'] = np.where(
        no_income_mask,
        df['DebtRatio'],
        df['DebtRatio'] * df['MonthlyIncome']
    )

    # 3. Drop DebtRatio - MonthlyDebt (combined with MonthlyIncome, which
    # stays in the dataset) carries the same information, so keeping both
    # is redundant and DebtRatio was the source of unstable / extreme
    # values after imputation.
    df = df.drop(columns=['DebtRatio'])

    print("Created 'MonthlyDebt' and dropped 'DebtRatio'.")
    return df

def engineer_credit_lines(df):
    """
    Extracts non-real estate loans (consumer loans/credit cards) 
    to understand the borrower's reliance on unsecured debt.
    """
    print("--- Engineering Credit Lines Features ---")
    
    # Non-real estate loans = Total loans - Real estate loans
    df['NonRealEstateLoans'] = df['NumberOfOpenCreditLinesAndLoans'] - df['NumberRealEstateLoansOrLines']
    
    # Ensure we don't have negative values just in case of data anomalies
    df['NonRealEstateLoans'] = df['NonRealEstateLoans'].clip(lower=0)
    
    print("Created 'NonRealEstateLoans' feature.")
    return df

def run_feature_engineering_pipeline(df):
    """
    Executes all feature engineering steps in the correct order.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe.
        
    Returns:
        pd.DataFrame: The dataframe with new engineered features.
    """
    print("\nStarting Feature Engineering Pipeline...")
    initial_cols = df.shape[1]
 
    df = engineer_debt_features(df)
    df = engineer_credit_lines(df)
    
    final_cols = df.shape[1]
    print(f"Feature Engineering Complete. Created {final_cols - initial_cols} new features.\n")
    
    return df