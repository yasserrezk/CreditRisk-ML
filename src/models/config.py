"""Shared configuration for the credit-risk model training pipeline."""
from pathlib import Path

RANDOM_STATE = 42

TARGET = "SeriousDlqin2yrs"
DROP_COLS = ["Id", TARGET]

# PROJECT_ROOT resolves to the folder that contains /src, /data, /artifacts
# (this file lives at src/models/config.py, so go up two levels)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "processed_train.csv"
TEST_PATH = DATA_DIR / "processed_test.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_MODEL_PKL_PATH = ARTIFACTS_DIR / "best_model.pkl"
BEST_MODEL_JOBLIB_PATH = ARTIFACTS_DIR / "best_model.joblib"
COMPARISON_CSV_PATH = ARTIFACTS_DIR / "model_comparison.csv"
SUBMISSION_CSV_PATH = ARTIFACTS_DIR / "test_predictions.csv"

CV_N_SPLITS = 5

# sklearn's SVC (kernel SVM) scales roughly O(n^2)-O(n^3) and becomes
# impractically slow past tens of thousands of rows. The training set here
# has ~115k rows after the split, so SVM is fit on a stratified subsample
# capped at this size. Set to None to disable the cap and use the full
# training set (expect a very long runtime).
SVM_MAX_TRAIN_ROWS = 20000
