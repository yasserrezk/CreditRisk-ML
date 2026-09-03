"""Credit Risk – Model Training, Tuning & Evaluation pipeline.

Trains all four models, tunes the three that support it (Logistic
Regression, LightGBM, XGBoost — SVM is left at a sensible baseline; see
SVM_MAX_TRAIN_ROWS in config.py for why), picks the best model by test-set
ROC-AUC, scores the held-out real test set, and writes everything the
Streamlit dashboard needs to /artifacts so the dashboard never has to
retrain.

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
from src.models import lightgbm_model, logistic_regression, svm_model, xgboost_model
from src.models.tuning import make_cv, tune

MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_BUNDLE_PATH = ARTIFACTS_DIR / "eval_bundle.joblib"


def _subsample_for_svm(X_train, y_train, max_rows, random_state):
    """Stratified subsample so kernel SVM training stays fast on large data."""
    if max_rows is None or len(X_train) <= max_rows:
        return X_train, y_train
    frac = max_rows / len(X_train)
    X_sub, _, y_sub, _ = train_test_split_data(X_train, y_train, test_size=1 - frac, random_state=random_state)
    print(f"SVM: subsampling training set from {len(X_train)} to {len(X_sub)} rows "
          f"(see SVM_MAX_TRAIN_ROWS in config.py).")
    return X_sub, y_sub


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_store = {}
    fitted_models = {}
    proba_map = {}

    # 1) Load + split
    df_train_full = load_train_full(TRAIN_PATH)
    X, y, feature_names = split_features_target(df_train_full)
    print("Features:", feature_names)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    cv = make_cv()

    # 2) Logistic Regression — baseline + tuned
    logreg_pipe = logistic_regression.build_baseline(RANDOM_STATE)
    logreg_pipe.fit(X_train, y_train)
    evaluate_model("Logistic Regression (baseline)", logreg_pipe, X_test, y_test, verbose=False)

    search_lr = tune(
        logistic_regression.build_search_estimator(RANDOM_STATE),
        logistic_regression.get_param_dist(),
        X_train, y_train, cv, n_iter=20,
    )
    print("Best params (LR):", search_lr.best_params_)
    best_logreg = search_lr.best_estimator_
    lr_metrics, lr_proba = evaluate_model("Logistic Regression", best_logreg, X_test, y_test, verbose=False)
    results_store["Logistic Regression"] = lr_metrics
    fitted_models["Logistic Regression"] = best_logreg
    proba_map["Logistic Regression"] = lr_proba

    # 3) LightGBM — baseline + tuned
    lgbm_baseline = lightgbm_model.build_baseline(RANDOM_STATE)
    lgbm_baseline.fit(X_train, y_train)
    evaluate_model("LightGBM (baseline)", lgbm_baseline, X_test, y_test, verbose=False)

    search_lgbm = tune(
        lightgbm_model.build_search_estimator(RANDOM_STATE),
        lightgbm_model.get_param_dist(),
        X_train, y_train, cv, n_iter=25,
    )
    print("Best params (LightGBM):", search_lgbm.best_params_)
    best_lgbm = search_lgbm.best_estimator_
    lgbm_metrics, lgbm_proba = evaluate_model("LightGBM", best_lgbm, X_test, y_test, verbose=False)
    results_store["LightGBM"] = lgbm_metrics
    fitted_models["LightGBM"] = best_lgbm
    proba_map["LightGBM"] = lgbm_proba

    # 4) XGBoost — baseline + tuned
    xgb_baseline = xgboost_model.build_baseline(RANDOM_STATE, y_train)
    xgb_baseline.fit(X_train, y_train)
    evaluate_model("XGBoost (baseline)", xgb_baseline, X_test, y_test, verbose=False)

    search_xgb = tune(
        xgboost_model.build_search_estimator(RANDOM_STATE),
        xgboost_model.get_param_dist(y_train),
        X_train, y_train, cv, n_iter=25,
    )
    print("Best params (XGBoost):", search_xgb.best_params_)
    best_xgb = search_xgb.best_estimator_
    xgb_metrics, xgb_proba = evaluate_model("XGBoost", best_xgb, X_test, y_test, verbose=False)
    results_store["XGBoost"] = xgb_metrics
    fitted_models["XGBoost"] = best_xgb
    proba_map["XGBoost"] = xgb_proba

    # 5) SVM — baseline only, on a subsample (kernel SVM doesn't scale to 100k+ rows)
    X_train_svm, y_train_svm = _subsample_for_svm(X_train, y_train, SVM_MAX_TRAIN_ROWS, RANDOM_STATE)
    svm_classifier = svm_model.build_baseline(RANDOM_STATE)
    svm_classifier.fit(X_train_svm, y_train_svm)
    svm_metrics, svm_proba = evaluate_model("SVM", svm_classifier, X_test, y_test, verbose=False)
    results_store["SVM"] = svm_metrics
    fitted_models["SVM"] = svm_classifier
    proba_map["SVM"] = svm_proba

    # 6) Compare all four models
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

    # 7) Feature importance / coefficients (for models that expose them)
    importances = {
        "Logistic Regression": pd.Series(
            best_logreg.named_steps["clf"].coef_[0], index=feature_names
        ).sort_values(),
        "LightGBM": pd.Series(best_lgbm.feature_importances_, index=feature_names).sort_values(),
        "XGBoost": pd.Series(best_xgb.feature_importances_, index=feature_names).sort_values(),
    }

    # 8) Save every fitted model + the best model + comparison table
    for name, model in fitted_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")

    joblib.dump(best_model, BEST_MODEL_PKL_PATH)
    joblib.dump(best_model, BEST_MODEL_JOBLIB_PATH)
    comparison_df.to_csv(COMPARISON_CSV_PATH)
    print(f"Best model saved to: {BEST_MODEL_PKL_PATH}")

    # 9) Everything the dashboard needs, precomputed (no retraining required)
    eval_bundle = {
        "y_test": y_test.to_numpy(),
        "proba_map": proba_map,
        "comparison_df": comparison_df,
        "feature_names": feature_names,
        "importances": importances,
        "best_model_name": best_model_name,
    }
    joblib.dump(eval_bundle, EVAL_BUNDLE_PATH)
    print(f"Dashboard eval bundle saved to: {EVAL_BUNDLE_PATH}")

    # 10) Score the real (held-out) test set with the best model
    df_test_real = load_test_real(TEST_PATH)
    X_test_real = df_test_real.drop(columns=[c for c in DROP_COLS if c in df_test_real.columns])
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
    submission.to_csv(SUBMISSION_CSV_PATH, index=False)
    print(f"Predicted default rate on real test set: {round(submission['predicted_class'].mean(), 4)}")
    print(f"Predictions saved to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    main()
