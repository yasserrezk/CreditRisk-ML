import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

NAME = "SVM"


def build_baseline(random_state):
    return CalibratedClassifierCV(
        SVC(class_weight="balanced", random_state=random_state, verbose=True),
        n_jobs=-1
    )


def build_search_estimator(random_state):
    return CalibratedClassifierCV(
        SVC(class_weight="balanced", random_state=random_state, verbose=True),
        n_jobs=-1
    )


def get_param_dist():
    return {
        "estimator__C": np.logspace(-3, 2, 20), 
        "estimator__kernel": ["linear", "rbf", "poly"],
        "estimator__gamma": ["scale", "auto"],
    }