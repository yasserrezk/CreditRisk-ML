import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

NAME = "SVM"


def build_baseline(random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=random_state)
            )
        ),
    ])


def build_search_estimator(random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=random_state)
            )
        ),
    ])


def get_param_dist():
    return {
        "clf__estimator__C": np.logspace(-3, 2, 20), 
        "clf__estimator__kernel": ["linear", "rbf", "poly"],
        "clf__estimator__gamma": ["scale", "auto"],
    }