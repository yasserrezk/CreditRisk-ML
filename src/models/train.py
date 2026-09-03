import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.model import logistic_regression, svm_model, xgboost_model
from src.models.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_JOBLIB_PATH,
    BEST_MODEL_PKL_PATH,
    COMPARISON_CSV_PATH,
    DROP_COLS,
    RANDOM_STATE,
    SUBMISSION_CSV_PATH,
    TEST_PATH,
    TRAIN_PATH,
)

from src.models.data_loader import (
    load_test_real,
    load_train_full,
    split_features_target,
    train_test_split_data,
)

from src.models.evaluation import evaluate_model
from src.models.tuning import make_cv, tune

MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_BUNDLE_PATH = ARTIFACTS_DIR / "eval_bundle.joblib"


def _unwrap_estimator(est):
    """Return the underlying estimator for pipeline-like or wrapper objects."""
    clf = est
    if hasattr(est, "named_steps"):
        clf = est.named_steps.get("clf", est)
    elif hasattr(est, "clf"):
        clf = getattr(est, "clf")

    if hasattr(clf, "estimator"):
        clf = clf.estimator
    if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
        clf = clf.calibrated_classifiers_[0].estimator
    return clf


def _get_coef(est):
    """Safely get coef_ from an estimator or a Pipeline-like object."""
    clf = _unwrap_estimator(est)
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    raise AttributeError(f"Estimator {type(clf).__name__} has no coef_")


def _get_feature_importances(est):
    """Safely get feature_importances_ from an estimator or Pipeline-like object.
    Falls back to coef_ if feature_importances_ is not present.
    """
    clf = _unwrap_estimator(est)

    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    raise AttributeError("Estimator has no feature_importances_ or coef_")


def _safe_feature_importance_series(est, feature_names):
    """Return feature importances or a zero vector when the estimator has none."""
    try:
        values = _get_feature_importances(est)
    except AttributeError:
        values = np.zeros(len(feature_names))
    return pd.Series(values, index=feature_names).sort_values()


def _get_model_path(name):
    """Return the saved path for a model."""
    safe_name = name.lower().replace(" ", "_")
    return MODELS_DIR / f"{safe_name}.joblib"


def _save_model(name, model):
    """Save a fitted model."""
    model_path = _get_model_path(name)
    joblib.dump(model, model_path)
    print(f"{name} saved to: {model_path}")


def _load_model(name):
    """Load a previously saved model."""
    model_path = _get_model_path(name)

    if model_path.exists():
        print(f"{name} already exists. Loading: {model_path}")
        return joblib.load(model_path)

    return None


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_store = {}
    fitted_models = {}
    proba_map = {}

    # ============================================================
    # 1) Load + split
    # ============================================================
    df_train_full = load_train_full(TRAIN_PATH)

    X, y, feature_names = split_features_target(df_train_full)

    print("Features:", feature_names)

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        random_state=RANDOM_STATE,
    )

    cv = make_cv()

    # ============================================================
    # 2) Logistic Regression
    # ============================================================
    print("\n===== Logistic Regression =====")

    best_logreg = _load_model("Logistic Regression")

    if best_logreg is None:
        print("Training Logistic Regression baseline...")

        logreg_pipe = logistic_regression.build_baseline(RANDOM_STATE)

        logreg_pipe.fit(X_train, y_train)

        evaluate_model(
            "Logistic Regression (baseline)",
            logreg_pipe,
            X_test,
            y_test,
            verbose=False,
        )

        print("Tuning Logistic Regression...")

        search_lr = tune(
            logistic_regression.build_search_estimator(RANDOM_STATE),
            logistic_regression.get_param_dist(),
            X_train,
            y_train,
            cv,
            n_iter=20,
        )

        print("Best params (LR):", search_lr.best_params_)

        best_logreg = search_lr.best_estimator_

        _save_model("Logistic Regression", best_logreg)

    lr_metrics, lr_proba = evaluate_model(
        "Logistic Regression",
        best_logreg,
        X_test,
        y_test,
        verbose=False,
    )

    results_store["Logistic Regression"] = lr_metrics
    fitted_models["Logistic Regression"] = best_logreg
    proba_map["Logistic Regression"] = lr_proba

    # ============================================================
    # 3) XGBoost
    # ============================================================
    print("\n===== XGBoost =====")

    best_xgb = _load_model("XGBoost")

    if best_xgb is None:
        print("Training XGBoost baseline...")

        xgb_baseline = xgboost_model.build_baseline(
            RANDOM_STATE,
            y_train,
        )

        xgb_baseline.fit(X_train, y_train)

        evaluate_model(
            "XGBoost (baseline)",
            xgb_baseline,
            X_test,
            y_test,
            verbose=False,
        )

        print("Tuning XGBoost...")

        search_xgb = tune(
            xgboost_model.build_search_estimator(RANDOM_STATE),
            xgboost_model.get_param_dist(y_train),
            X_train,
            y_train,
            cv,
            n_iter=15,
        )

        print("Best params (XGBoost):", search_xgb.best_params_)

        best_xgb = search_xgb.best_estimator_

        _save_model("XGBoost", best_xgb)

    xgb_metrics, xgb_proba = evaluate_model(
        "XGBoost",
        best_xgb,
        X_test,
        y_test,
        verbose=False,
    )

    results_store["XGBoost"] = xgb_metrics
    fitted_models["XGBoost"] = best_xgb
    proba_map["XGBoost"] = xgb_proba

    # ============================================================
    # 4) SVM
    # ============================================================
    print("\n===== SVM =====")

    svm_classifier = _load_model("SVM")

    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train[:80000], y_train[:80000])
    X_test_scaled = std.transform(X_test)

    if svm_classifier is None:
        print("Training SVM baseline...")

        svm_classifier = svm_model.build_baseline(RANDOM_STATE)

        svm_classifier.fit(
            X_train_scaled,
            y_train[:80000],
        )

        _save_model("SVM", svm_classifier)

    svm_metrics, svm_proba = evaluate_model(
        "SVM",
        svm_classifier,
        X_test_scaled,
        y_test,
        verbose=False,
    )

    results_store["SVM"] = svm_metrics
    fitted_models["SVM"] = svm_classifier
    proba_map["SVM"] = svm_proba

    # ============================================================
    # 5) Compare models
    # ============================================================
    comparison_df = (
        pd.DataFrame(list(results_store.values()))
        .set_index("model")
        .round(4)
        .sort_values("roc_auc", ascending=False)
    )

    print("\n===== Model comparison (sorted by ROC-AUC) =====")
    print(comparison_df)

    best_model_name = comparison_df["roc_auc"].idxmax()
    best_model = fitted_models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # ============================================================
    # 6) Feature importance / coefficients
    # ============================================================

    importances = {
        "Logistic Regression": pd.Series(
            _get_coef(best_logreg),
            index=feature_names,
        ).sort_values(),
        "XGBoost": _safe_feature_importance_series(best_xgb, feature_names),
        "SVM": _safe_feature_importance_series(fitted_models["SVM"], feature_names),
    }

    # ============================================================
    # 7) Save best model
    # ============================================================
    joblib.dump(best_model, BEST_MODEL_PKL_PATH)
    joblib.dump(best_model, BEST_MODEL_JOBLIB_PATH)

    comparison_df.to_csv(COMPARISON_CSV_PATH)

    print(f"Best model saved to: {BEST_MODEL_PKL_PATH}")

    # ============================================================
    # 8) Save evaluation bundle
    # ============================================================
    eval_bundle = {
        "y_test": y_test.to_numpy(),
        "proba_map": proba_map,
        "comparison_df": comparison_df,
        "feature_names": feature_names,
        "importances": importances,
        "best_model_name": best_model_name,
    }

    joblib.dump(
        eval_bundle,
        EVAL_BUNDLE_PATH,
    )

    print(f"Dashboard eval bundle saved to: {EVAL_BUNDLE_PATH}")

    # ============================================================
    # 9) Score real held-out test set
    # ============================================================
    df_test_real = load_test_real(TEST_PATH)

    X_test_real = df_test_real.drop(
        columns=[c for c in DROP_COLS if c in df_test_real.columns]
    )

    assert list(X_test_real.columns) == feature_names

    test_proba = best_model.predict_proba(X_test_real)[:, 1]

    test_pred = (test_proba >= 0.5).astype(int)

    assert not np.isnan(test_proba).any()

    submission = pd.DataFrame(
        {
            "Id": df_test_real["Id"],
            "predicted_probability": test_proba,
            "predicted_class": test_pred,
        }
    )

    submission.to_csv(
        SUBMISSION_CSV_PATH,
        index=False,
    )

    print(
        "Predicted default rate on real test set:",
        round(submission["predicted_class"].mean(), 4),
    )

    print(f"Predictions saved to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    main()
