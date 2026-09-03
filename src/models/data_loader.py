"""Load the processed credit-risk datasets and produce the train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.config import DROP_COLS, RANDOM_STATE, TARGET, TEST_PATH, TRAIN_PATH


def load_train_full(path=TRAIN_PATH):
    """Load the full processed training set."""
    df_train_full = pd.read_csv(path)
    print("Shape:", df_train_full.shape)
    print("\nTarget distribution:")
    print(df_train_full[TARGET].value_counts(normalize=True).rename("ratio"))
    return df_train_full


def load_test_real(path=TEST_PATH):
    """Load the held-out (real) test set used for the final submission."""
    return pd.read_csv(path)


def split_features_target(df_train_full, drop_cols=DROP_COLS, target=TARGET):
    """Split a dataframe into feature matrix X and target vector y."""
    X = df_train_full.drop(columns=drop_cols)
    y = df_train_full[target]
    feature_names = X.columns.tolist()
    return X, y, feature_names


def train_test_split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split, matching the original notebook's setup."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
