from time import time

import pandas as pd
from pathlib import Path

# 1. Define the project root directory dynamically
# Assuming this script is inside src/data/data_fetching/
# .parents[3] navigates up 4 levels to reach the 'CreditRisk-ML' root folder
project_root = Path(__file__).resolve().parents[3]

# 2. Construct the exact paths for the raw data files
train_path = project_root / 'data' / 'raw' / 'training.csv'
test_path = project_root / 'data' / 'raw' / 'test.csv'

def load_raw_data():
    """
    Loads the raw training and testing datasets from the data/raw directory.
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames.
    """
    print(f"Loading training data from: {train_path}")
    print(f"Loading testing data from: {test_path}")
    
    # Load the CSV files into pandas DataFrames
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print("Data loaded successfully!\n")
        return train_data, test_data
        
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the paths.\n{e}")
        return None, None

processed_train_path = project_root / 'data' / 'processed' / 'processed_train.csv'
processed_test_path = project_root / 'data' / 'processed' / 'processed_test.csv'

def load_processed_data():
    """
    Loads the processed training and testing datasets from the data/processed directory.
    
    Returns:
        tuple: (processed_train, processed_test) as pandas DataFrames.
    """
    print(f"Loading training data from: {processed_train_path}")
    print(f"Loading testing data from: {processed_test_path}")
    
    # Load the CSV files into pandas DataFrames
    try:
        processed_train = pd.read_csv(processed_train_path)
        processed_test = pd.read_csv(processed_test_path)
        print("Data loaded successfully!\n")
        return processed_train, processed_test
        
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the paths.\n{e}")
        return None, None
 
if __name__ == "__main__":
    # Execute the function and unpack the returned DataFrames
    train_df, test_df = load_raw_data()
    
    if train_df is not None and test_df is not None:
        # Display basic information to verify successful loading
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")

    # Execute the function and unpack the returned DataFrames
    processed_train_df, processed_test_df = load_processed_data()

    if processed_train_df is not None and processed_test_df is not None:
        # Display basic information to verify successful loading
        print(f"Processed training data shape: {processed_train_df.shape}")
        print(f"Processed testing data shape: {processed_test_df.shape}")