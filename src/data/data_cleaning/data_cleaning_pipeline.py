import pandas as pd
import numpy as np

def rename_id_column(df):
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Id'})
    return df


def clean_duplicates(df):
    cols_to_check = [col for col in df.columns if col.lower() != 'id']
    duplicates_count = df.duplicated(subset=cols_to_check).sum()
    print(f"Found {duplicates_count} duplicate rows (ignoring ID).")
    df = df.drop_duplicates(subset=cols_to_check, keep='first')
    print("Duplicate rows removed successfully.")
    return df

def clean_age(df, is_train=True):
    if is_train:
        df = df[(df['age'] >= 21) | (df['age'].isna())]
    else:
        df['age'] = df['age'].clip(lower=21)
    return df

def clean_revolving(df, is_train=True):
    if is_train:
        df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]
    else:
        df['RevolvingUtilizationOfUnsecuredLines'] = (
            df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)
        )
    return df

def clean_number_of_delays(df, is_train=True):
    cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    if is_train:
        for c in cols:
            df = df[(df[c] <= 20) | (df[c].isna())]
    else:
        for c in cols:
            df[c] = df[c].clip(upper=20)
    return df

def run_cleaning_pipeline(df, is_train=True):
    print("--- Starting Data Cleaning Pipeline ---")
    initial_shape = df.shape

    df = rename_id_column(df)

    if is_train:
        df = clean_duplicates(df)

    df = clean_revolving(df, is_train=is_train)
    df = clean_number_of_delays(df, is_train=is_train)

    final_shape = df.shape
    print(f"\n--- Cleaning Complete ---")
    print(f"Initial Shape: {initial_shape}")
    print(f"Final Shape: {final_shape}")
    print(f"Total rows removed: {initial_shape[0] - final_shape[0]}")

    return df