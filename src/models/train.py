"""Credit Risk – Model Training, Tuning & Evaluation pipeline.

Trains four models and saves each fitted model immediately.
On re-run, an existing saved model is loaded instead of retraining it.

Run from the project root:
    python -m src.models.train
"""

import joblib
import numpy as np
import pandas as pd

from src.models.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_JOBLIB_PATH,
    BEST_MODEL_PKL_PATH,
    COMPARISON_CSV_PATH,
    DROP_COLS,
    RANDOM_STATE,
    SUBMISSION_CSV_PATH,
    SVM_MAX_TRAIN_ROWS,
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
from src.models import logistic_regression, svm_model, xgboost_model
from src.models.tuning import make_cv, tune


MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_BUNDLE_PATH = ARTIFACTS_DIR / "eval_bundle.joblib"


def _subsample_for_svm(X_train, y_train, max_rows, random_state):
    """Stratified subsample so kernel SVM training stays fast."""
    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train

    frac = max_rows / len(X_train)

    X_sub, _, y_sub, _ = train_test_split_data(
        X_train,
        y_train,
        test_size=1 - frac,
        random_state=random_state,
    )

    print(
        f"SVM: subsampling training set from {len(X_train)} "
        f"to {len(X_sub)} rows."
    )

    return X_sub, y_sub


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

    if svm_classifier is None:
        print("Training SVM baseline...")

        X_train_svm, y_train_svm = _subsample_for_svm(
            X_train,
            y_train,
            SVM_MAX_TRAIN_ROWS,
            RANDOM_STATE,
        )

        svm_classifier = svm_model.build_baseline(RANDOM_STATE)

        svm_classifier.fit(
            X_train_svm,
            y_train_svm,
        )

        _save_model("SVM", svm_classifier)

    svm_metrics, svm_proba = evaluate_model(
        "SVM",
        svm_classifier,
        X_test,
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
            best_logreg.named_steps["clf"].coef_[0],
            index=feature_names,
        ).sort_values(),

        "XGBoost": pd.Series(
            best_xgb.feature_importances_,
            index=feature_names,
        ).sort_values(),
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
        columns=[
            c for c in DROP_COLS
            if c in df_test_real.columns
        ]
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

    print(
        f"Predictions saved to: {SUBMISSION_CSV_PATH}"
    )


if __name__ == "__main__":
    main()