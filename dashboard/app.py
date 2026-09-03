"""Credit Risk model evaluation dashboard.

Reads the artifacts written by `python -m src.models.train` — no retraining
happens here, so the dashboard loads instantly.

Run from the project root:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import ARTIFACTS_DIR, COMPARISON_CSV_PATH, SUBMISSION_CSV_PATH  # noqa: E402

EVAL_BUNDLE_PATH = ARTIFACTS_DIR / "eval_bundle.joblib"

st.set_page_config(page_title="Credit Risk — Model Evaluation", layout="wide")


@st.cache_data
def load_artifacts():
    if not EVAL_BUNDLE_PATH.exists():
        return None
    bundle = joblib.load(EVAL_BUNDLE_PATH)
    submission = pd.read_csv(SUBMISSION_CSV_PATH) if SUBMISSION_CSV_PATH.exists() else None
    return bundle, submission


data = load_artifacts()

st.title("Credit Risk — Model Evaluation")

if data is None:
    st.error(
        "No trained artifacts found. Run `python -m src.models.train` from the "
        "project root first, then reload this page."
    )
    st.stop()

bundle, submission = data
y_test = bundle["y_test"]
proba_map = bundle["proba_map"]
comparison_df = bundle["comparison_df"]
feature_names = bundle["feature_names"]
importances = bundle["importances"]
best_model_name = bundle["best_model_name"]

model_names = list(proba_map.keys())

# ---- Top-line summary ----
best_row = comparison_df.loc[best_model_name]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best model", best_model_name)
col2.metric("ROC-AUC", f"{best_row['roc_auc']:.4f}")
col3.metric("F1", f"{best_row['f1']:.4f}")
col4.metric("Recall", f"{best_row['recall']:.4f}")

st.divider()

# ---- Model comparison table ----
st.subheader("Model comparison")
st.dataframe(
    comparison_df.style.highlight_max(subset=["roc_auc", "f1", "recall", "precision", "accuracy"], color="#d4f7d4"),
    width="stretch",
)

metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
bar_df = comparison_df[metric_cols].reset_index().melt(id_vars="model", var_name="metric", value_name="score")
fig_bar = px.bar(bar_df, x="metric", y="score", color="model", barmode="group", range_y=[0, 1])
st.plotly_chart(fig_bar, width="stretch")

st.divider()

# ---- ROC / PR curves ----
st.subheader("ROC & Precision-Recall curves")
c1, c2 = st.columns(2)

with c1:
    fig_roc = go.Figure()
    for name, proba in proba_map.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = comparison_df.loc[name, "roc_auc"]
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Random"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, width="stretch")

with c2:
    fig_pr = go.Figure()
    for name, proba in proba_map.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=name))
    baseline_rate = float(np.mean(y_test))
    fig_pr.add_hline(y=baseline_rate, line_dash="dash", line_color="grey",
                      annotation_text=f"Baseline ({baseline_rate:.3f})")
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, width="stretch")

st.divider()

# ---- Per-model inspection: confusion matrix + threshold ----
st.subheader("Inspect a model")
selected_model = st.selectbox("Model", model_names, index=model_names.index(best_model_name))
threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)

proba = proba_map[selected_model]
y_pred = (proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

c3, c4 = st.columns([1, 1])
with c3:
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        title=f"{selected_model} — Confusion Matrix (threshold={threshold:.2f})",
    )
    st.plotly_chart(fig_cm, width="stretch")

with c4:
    if selected_model in importances:
        imp = importances[selected_model]
        label = "Coefficient" if selected_model == "Logistic Regression" else "Importance"
        fig_imp = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            labels={"x": label, "y": "Feature"},
            title=f"{selected_model} — Feature {label}",
        )
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info(f"{selected_model} does not expose feature importances.")

st.divider()

# ---- Predictions on the real held-out test set ----
st.subheader("Predictions on the real test set")
if submission is not None:
    st.caption(f"Scored with the best model ({best_model_name}). {len(submission)} rows.")
    default_rate = submission["predicted_class"].mean()
    st.metric("Predicted default rate", f"{default_rate:.2%}")
    st.dataframe(submission.head(200), width="stretch")
    st.download_button(
        "Download full predictions (CSV)",
        submission.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("No test-set predictions found. Run the training pipeline to generate them.")
