#!/usr/bin/env bash
# Runs the model training pipeline, then verifies the expected
# pkl/joblib artifacts were actually produced.
#
# Usage (from anywhere): ./run_training.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
MODELS_DIR="$ARTIFACTS_DIR/models"

REQUIRED_FILES=(
    "$ARTIFACTS_DIR/best_model.pkl"
    "$ARTIFACTS_DIR/best_model.joblib"
    "$ARTIFACTS_DIR/eval_bundle.joblib"
    "$ARTIFACTS_DIR/model_comparison.csv"
    "$ARTIFACTS_DIR/test_predictions.csv"
    "$MODELS_DIR/logistic_regression.joblib"
    "$MODELS_DIR/lightgbm.joblib"
    "$MODELS_DIR/xgboost.joblib"
    "$MODELS_DIR/svm.joblib"
)

echo "[1/2] Running training pipeline..."
if ! python3 -m src.models.train; then
    echo "ERROR: training pipeline exited with an error." >&2
    exit 1
fi

echo
echo "[2/2] Verifying artifacts..."
missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        size=$(ls -lh "$f" | awk '{print $5}')
        echo "  OK   ($size)  $f"
    else
        echo "  MISSING        $f"
        missing=1
    fi
done

echo
if [[ "$missing" -eq 1 ]]; then
    echo "FAILED: one or more expected pkl/joblib artifacts were not created." >&2
    exit 1
fi

echo "SUCCESS: all model artifacts present."
exit 0
