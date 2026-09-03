from pathlib import Path

RANDOM_STATE = 42

TARGET = "SeriousDlqin2yrs"
DROP_COLS = ["Id", TARGET]

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

