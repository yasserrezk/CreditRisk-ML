"""Logistic Regression model: baseline pipeline + tuning search space."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Logistic Regression"


def build_pipeline(random_state, max_iter=2000):
    """Scaler + LogisticRegression pipeline used for both baseline and tuning."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_baseline(random_state):
    return build_pipeline(random_state, max_iter=2000)


def build_search_estimator(random_state):
    return build_pipeline(random_state, max_iter=3000)


def get_param_dist():
    return {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear", "saga"],
    }
