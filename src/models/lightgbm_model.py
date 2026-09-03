"""LightGBM model: baseline classifier + tuning search space."""
from lightgbm import LGBMClassifier

NAME = "LightGBM"


def build_baseline(random_state, n_estimators=200):
    return LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=n_estimators,
        verbose=-1,
    )


def build_search_estimator(random_state):
    return LGBMClassifier(random_state=random_state, class_weight="balanced", verbose=-1)


def get_param_dist():
    return {
        "n_estimators": [100, 200, 300, 500],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1, 5],
        "reg_lambda": [0, 0.1, 1, 5],
    }
