"""XGBoost model: baseline classifier + reduced tuning search space.

Uses `scale_pos_weight` to handle class imbalance.
"""

from xgboost import XGBClassifier


NAME = "XGBoost"


def compute_scale_pos_weight(y_train):
    """Compute the negative-to-positive class ratio."""
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    if n_positive == 0:
        raise ValueError("Training data contains no positive samples.")

    return n_negative / n_positive


def build_baseline(random_state, y_train, n_estimators=200):
    """Build the baseline XGBoost classifier."""
    scale_pos_weight = compute_scale_pos_weight(y_train)

    return XGBClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
    )


def build_search_estimator(random_state):
    """Build the base estimator for RandomizedSearchCV."""
    return XGBClassifier(
        random_state=random_state,
        eval_metric="auc",
    )


def get_param_dist(y_train):
    """Return a reduced hyperparameter search space."""
    scale_pos_weight = compute_scale_pos_weight(y_train)

    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "reg_alpha": [0, 1],
        "reg_lambda": [1, 2],
        "scale_pos_weight": [
            1,
            scale_pos_weight,
        ],
    }