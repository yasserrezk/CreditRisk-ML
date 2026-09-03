"""XGBoost model: baseline classifier + tuning search space.

Unlike the other models, XGBoost's baseline and its search space both need
`scale_pos_weight`, which is derived from the training target's class
imbalance -- so those two functions take `y_train` as an argument.
"""
from xgboost import XGBClassifier

NAME = "XGBoost"


def compute_scale_pos_weight(y_train):
    return (y_train == 0).sum() / (y_train == 1).sum()


def build_baseline(random_state, y_train, n_estimators=200):
    scale_pos_weight = compute_scale_pos_weight(y_train)
    return XGBClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        eval_metric="auc",
        use_label_encoder=False,
    )


def build_search_estimator(random_state):
    return XGBClassifier(random_state=random_state, eval_metric="auc", use_label_encoder=False)


def get_param_dist(y_train):
    scale_pos_weight = compute_scale_pos_weight(y_train)
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.5, 1],
        "reg_alpha": [0, 0.1, 1, 5],
        "reg_lambda": [0.5, 1, 2, 5],
        "scale_pos_weight": [1, scale_pos_weight / 2, scale_pos_weight],
    }
