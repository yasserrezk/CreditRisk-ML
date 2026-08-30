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

if __name__ == "__main__":
    # Execute the function and unpack the returned DataFrames
    train_df, test_df = load_raw_data()
    
    if train_df is not None and test_df is not None:
        # Display basic information to verify successful loading
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")