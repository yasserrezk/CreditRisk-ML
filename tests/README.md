# Tests

This directory is reserved for automated tests using pytest.

Planned coverage includes:

- Data-cleaning rules and duplicate handling.
- Feature-engineering calculations.
- Imputation behavior and train/test separation.
- API request schemas and endpoint responses once the API exists.
- Model output shape and probability constraints once models are added.

Tests should use small fixtures rather than modifying the immutable files in
`data/raw/`. Run the test suite from the repository root with:

```bash
pytest
```

No test modules are currently present.