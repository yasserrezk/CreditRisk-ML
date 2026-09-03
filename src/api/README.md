# API layer

This directory is reserved for the FastAPI backend that will expose the credit
risk model for real-time predictions.

## Planned contents

- `main.py`: FastAPI application and prediction endpoints.
- `schemas.py`: Request and response validation models.
- `services.py`: Model loading, preprocessing, and prediction services.

## Expected flow

1. Receive validated borrower features.
2. Apply the same preprocessing used to create the processed datasets.
3. Load a serialized model and generate a risk probability.
4. Return a documented response containing the prediction and probability.

No API implementation is currently present in this directory. Keep model
loading and preprocessing consistent with `src/features/` before adding
production endpoints.