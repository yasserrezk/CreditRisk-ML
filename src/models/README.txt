# Modeling layer

This directory is reserved for model training and evaluation code.

## Planned responsibilities

- Load `data/processed/processed_train.csv`.
- Separate the `SeriousDlqin2yrs` target from predictor columns.
- Train and cross-validate credit-risk classifiers.
- Report metrics appropriate for imbalanced classification, such as ROC-AUC,
  PR-AUC, recall, precision, and confusion matrices.
- Serialize the selected model and preprocessing objects into `artifacts/`.

The processed test file has no target labels, so it should only be used for
inference or submission generation. Model code should reuse the feature
definitions in `src/features/` and avoid fitting transformations on test data.

No model implementation is currently present in this directory.
