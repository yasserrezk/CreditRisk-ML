"""Generic RandomizedSearchCV wrapper used to tune every model the same way."""
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.models.config import CV_N_SPLITS, RANDOM_STATE


def make_cv(n_splits=CV_N_SPLITS, random_state=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune(estimator, param_dist, X_train, y_train, cv, n_iter=20, scoring="roc_auc", random_state=RANDOM_STATE):
    """Run RandomizedSearchCV and return the fitted search object."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search
