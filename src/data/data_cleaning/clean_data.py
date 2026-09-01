import os
import sys
from pathlib import Path

# Setup paths to allow importing from other modules
project_root = Path.cwd()
while not (project_root / 'src').exists() and project_root.parent != project_root:
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import modules from our MLOps structure
from src.data.data_fetching.load_data import load_raw_data
from src.data.data_cleaning.data_cleaning_pipeline import (
    clean_duplicates,
    clean_age,
    clean_revolving,
    clean_number_of_delays,
    run_cleaning_pipeline
)
from src.features.feature_engineering import engineer_credit_lines, engineer_debt_features
from src.features.smart_imputation import (
    run_imputation_pipeline,
    create_imputer
)

def process_dataset(df, dataset_name="Dataset", imputer=None, fit_imputer=False, is_train=True):
    print(f"\n--- Processing {dataset_name} ---")

    df = run_cleaning_pipeline(df, is_train=is_train)
    df = engineer_debt_features(df)
    df = run_imputation_pipeline(df, imputer=imputer, fit=fit_imputer)
    df = engineer_credit_lines(df)

    # Final safety-net cleaning pass: age, revolving utilization and delay
    # counts are all part of smart_cols in the imputer, so the imputer
    # could have produced values that break the original business rules.
    # Re-apply the same checks after imputation.
    df = clean_age(df, is_train=is_train)
    df = clean_revolving(df, is_train=is_train)
    df = clean_number_of_delays(df, is_train=is_train)

    if is_train:
        df = clean_duplicates(df)

    return df


if __name__ == "__main__":
    print("--- Starting Full Data Preparation Pipeline ---")

    # 1. Load raw data
    train_data, test_data = load_raw_data()

    # 2. Create ONE imputer
    imputer = create_imputer()

    # Setup processed directory
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if train_data is not None:

        # Process Training Data
        processed_train = process_dataset(train_data, dataset_name="Training Data", imputer=imputer, fit_imputer=True, is_train=True)

        train_save_path = processed_dir / 'processed_train.csv'
        processed_train.to_csv(train_save_path, index=False)

        print(
            f"Final processed training data saved successfully to: "
            f"{train_save_path}"
        )

    if test_data is not None:

        # Process Test Data using the SAME fitted imputer
        processed_test = process_dataset(test_data, dataset_name="Test Data", imputer=imputer, fit_imputer=False, is_train=False)

        test_save_path = processed_dir / 'processed_test.csv'
        processed_test.to_csv(test_save_path, index=False)

        print(
            f"Final processed test data saved successfully to: "
            f"{test_save_path}"
        )

    print("\nAll Pipelines Executed Successfully!")