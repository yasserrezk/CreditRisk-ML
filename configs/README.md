# Configuration

This directory is reserved for project configuration files, such as:

- Model hyperparameters.
- Feature lists and preprocessing options.
- Classification thresholds.
- Environment-specific settings.

YAML or JSON files can be added here as the modeling and serving layers are
implemented. Keep credentials and other secrets out of this directory; use
environment variables or a secret manager instead.
