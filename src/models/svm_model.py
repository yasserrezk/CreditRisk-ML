"""SVM model: baseline classifier + tuning search space."""
import numpy as np
from sklearn.svm import SVC

NAME = "SVM"


def build_baseline(random_state):
    return SVC(
        probability=True,
        class_weight="balanced",
        random_state=random_state,
        kernel="rbf",
        gamma="scale",
        C=1.0,
    )


def build_search_estimator(random_state):
    return SVC(probability=True, class_weight="balanced", random_state=random_state)


def get_param_dist():
    return {
        "C": np.logspace(-3, 2, 20),
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    }
