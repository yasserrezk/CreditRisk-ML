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
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import (  # noqa: E402
    ARTIFACTS_DIR,
    COMPARISON_CSV_PATH,
    SUBMISSION_CSV_PATH,
)

EVAL_BUNDLE_PATH = ARTIFACTS_DIR / "eval_bundle.joblib"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk — Model Evaluation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# Midnight / Purple / Cyan
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(139, 92, 246, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 5% 25%,
                rgba(34, 211, 238, 0.07),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #0B1020 0%,
                #0F172A 50%,
                #0B1020 100%
            );

        color: #F8FAFC;
    }

    .main {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }


    /* ========================================================
       TITLES
       ======================================================== */

    h1 {
        color: #F8FAFC !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 0.3rem !important;
    }

    h2,
    h3 {
        color: #A78BFA !important;
        font-weight: 700 !important;
    }

    p,
    label,
    .stMarkdown {
        color: #CBD5E1 !important;
    }


    /* Gradient line under title */

    h1::after {
        content: "";
        display: block;

        width: 90px;
        height: 4px;

        margin-top: 12px;

        border-radius: 10px;

        background:
            linear-gradient(
                90deg,
                #8B5CF6,
                #22D3EE
            );
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border: none !important;
        height: 1px !important;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(139, 92, 246, 0.6),
                rgba(34, 211, 238, 0.4),
                transparent
            ) !important;

        margin: 1.8rem 0 !important;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(18, 26, 47, 0.96),
                rgba(15, 23, 42, 0.96)
            );

        border:
            1px solid rgba(139, 92, 246, 0.25);

        border-radius: 18px;

        padding: 20px 22px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);

        transition:
            transform 0.25s ease,
            border 0.25s ease,
            box-shadow 0.25s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);

        border-color:
            rgba(139, 92, 246, 0.65);

        box-shadow:
            0 15px 35px rgba(0, 0, 0, 0.40),
            0 0 25px rgba(139, 92, 246, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #22D3EE !important;
    }


    /* ========================================================
       SELECT BOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background: #121A2F !important;

        border:
            1px solid rgba(139, 92, 246, 0.35) !important;

        border-radius: 12px !important;

        color: #F8FAFC !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #8B5CF6 !important;

        box-shadow:
            0 0 0 1px rgba(139, 92, 246, 0.15);
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    div[data-testid="stSlider"] {
        padding: 10px 5px;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background-color: #22D3EE !important;

        border:
            2px solid #67E8F9 !important;

        box-shadow:
            0 0 12px rgba(34, 211, 238, 0.4);
    }

    div[data-testid="stSlider"]
    div[data-baseweb="slider"]
    > div:first-child {
        background-color: #334155 !important;
    }

    div[data-testid="stSlider"]
    div[data-baseweb="slider"]
    > div:nth-child(2) {
        background:
            linear-gradient(
                90deg,
                #8B5CF6,
                #22D3EE
            ) !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border:
            1px solid rgba(139, 92, 246, 0.25);

        border-radius: 14px;

        overflow: hidden;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.25);
    }


    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    .stDownloadButton > button {
        width: 100%;

        background:
            linear-gradient(
                135deg,
                #8B5CF6,
                #6366F1
            ) !important;

        color: white !important;

        border: none !important;

        border-radius: 12px !important;

        font-weight: 800 !important;

        padding: 0.7rem 1rem !important;

        transition:
            all 0.25s ease !important;
    }

    .stDownloadButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #A78BFA,
                #22D3EE
            ) !important;

        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(139, 92, 246, 0.3);
    }


    /* ========================================================
       GENERAL BUTTONS
       ======================================================== */

    .stButton > button {
        background: #121A2F !important;

        color: #A78BFA !important;

        border:
            1px solid rgba(139, 92, 246, 0.45) !important;

        border-radius: 10px !important;

        font-weight: 700 !important;
    }

    .stButton > button:hover {
        border-color: #22D3EE !important;
        color: #22D3EE !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        background:
            rgba(18, 26, 47, 0.9) !important;

        border:
            1px solid rgba(34, 211, 238, 0.25) !important;

        border-radius: 12px !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0B1020 0%,
                #111827 100%
            );

        border-right:
            1px solid rgba(139, 92, 246, 0.2);
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #A78BFA !important;
    }


    /* ========================================================
       PLOTLY CONTAINERS
       ======================================================== */

    div[data-testid="stPlotlyChart"] {
        background:
            rgba(15, 23, 42, 0.65);

        border:
            1px solid rgba(139, 92, 246, 0.18);

        border-radius: 16px;

        padding: 8px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.25);
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0B1020;
    }

    ::-webkit-scrollbar-thumb {
        background:
            linear-gradient(
                180deg,
                #8B5CF6,
                #22D3EE
            );

        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #A78BFA;
    }


    /* ========================================================
       HIDE STREAMLIT BRANDING
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_data
def load_artifacts():
    if not EVAL_BUNDLE_PATH.exists():
        return None

    bundle = joblib.load(EVAL_BUNDLE_PATH)

    submission = (
        pd.read_csv(SUBMISSION_CSV_PATH)
        if SUBMISSION_CSV_PATH.exists()
        else None
    )

    return bundle, submission


data = load_artifacts()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Credit Risk — Model Evaluation")


# ============================================================
# CHECK ARTIFACTS
# ============================================================

if data is None:
    st.error(
        "No trained artifacts found. "
        "Run `python -m src.models.train` from the "
        "project root first, then reload this page."
    )
    st.stop()


# ============================================================
# EXTRACT DATA
# ============================================================

bundle, submission = data

y_test = bundle["y_test"]
proba_map = bundle["proba_map"]
comparison_df = bundle["comparison_df"]
feature_names = bundle["feature_names"]
importances = bundle["importances"]
best_model_name = bundle["best_model_name"]

model_names = list(proba_map.keys())


# ============================================================
# TOP-LINE SUMMARY
# ============================================================

best_row = comparison_df.loc[best_model_name]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best Model",
    best_model_name,
)

col2.metric(
    "ROC-AUC",
    f"{best_row['roc_auc']:.4f}",
)

col3.metric(
    "F1 Score",
    f"{best_row['f1']:.4f}",
)

col4.metric(
    "Recall",
    f"{best_row['recall']:.4f}",
)


st.divider()


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

st.subheader("Model Comparison")

styled_comparison = (
    comparison_df.style
    .highlight_max(
        subset=[
            "roc_auc",
            "f1",
            "recall",
            "precision",
            "accuracy",
        ],
        color="#C4B5FD",
    )
)

st.dataframe(
    styled_comparison,
    width="stretch",
)


# ============================================================
# MODEL COMPARISON BAR CHART
# ============================================================

metric_cols = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

bar_df = (
    comparison_df[metric_cols]
    .reset_index()
    .melt(
        id_vars="model",
        var_name="metric",
        value_name="score",
    )
)

fig_bar = px.bar(
    bar_df,
    x="metric",
    y="score",
    color="model",
    barmode="group",
    range_y=[0, 1],
    color_discrete_sequence=[
        "#8B5CF6",
        "#22D3EE",
        "#A78BFA",
        "#6366F1",
        "#38BDF8",
    ],
)

fig_bar.update_layout(
    title=dict(
        text="Model Performance Comparison",
        font=dict(
            color="#F8FAFC",
            size=20,
        ),
    ),
    xaxis=dict(
        title="Metric",
        color="#CBD5E1",
        gridcolor="rgba(148,163,184,0.10)",
    ),
    yaxis=dict(
        title="Score",
        color="#CBD5E1",
        gridcolor="rgba(148,163,184,0.10)",
        range=[0, 1],
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        font=dict(
            color="#CBD5E1",
        ),
    ),
)

st.plotly_chart(
    fig_bar,
    width="stretch",
)


st.divider()


# ============================================================
# ROC + PRECISION RECALL
# ============================================================

st.subheader("ROC & Precision-Recall Curves")

c1, c2 = st.columns(2)


# ============================================================
# ROC CURVE
# ============================================================

with c1:

    fig_roc = go.Figure()

    roc_colors = [
        "#8B5CF6",
        "#22D3EE",
        "#A78BFA",
        "#6366F1",
        "#38BDF8",
    ]

    for i, (name, proba) in enumerate(proba_map.items()):

        fpr, tpr, _ = roc_curve(
            y_test,
            proba,
        )

        auc_val = comparison_df.loc[
            name,
            "roc_auc",
        ]

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(
                    width=3,
                    color=roc_colors[
                        i % len(roc_colors)
                    ],
                ),
            )
        )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash",
                color="#64748B",
            ),
            name="Random",
        )
    )

    fig_roc.update_layout(
        title=dict(
            text="ROC Curve",
            font=dict(
                color="#A78BFA",
                size=20,
            ),
        ),
        xaxis=dict(
            title="False Positive Rate",
            color="#CBD5E1",
            gridcolor="rgba(148,163,184,0.10)",
        ),
        yaxis=dict(
            title="True Positive Rate",
            color="#CBD5E1",
            gridcolor="rgba(148,163,184,0.10)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(
                color="#CBD5E1",
            ),
        ),
    )

    st.plotly_chart(
        fig_roc,
        width="stretch",
    )


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

with c2:

    fig_pr = go.Figure()

    pr_colors = [
        "#22D3EE",
        "#8B5CF6",
        "#38BDF8",
        "#A78BFA",
        "#6366F1",
    ]

    for i, (name, proba) in enumerate(proba_map.items()):

        prec, rec, _ = precision_recall_curve(
            y_test,
            proba,
        )

        fig_pr.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=name,
                line=dict(
                    width=3,
                    color=pr_colors[
                        i % len(pr_colors)
                    ],
                ),
            )
        )

    baseline_rate = float(
        np.mean(y_test)
    )

    fig_pr.add_hline(
        y=baseline_rate,
        line_dash="dash",
        line_color="#64748B",
        annotation_text=(
            f"Baseline ({baseline_rate:.3f})"
        ),
        annotation_font_color="#94A3B8",
    )

    fig_pr.update_layout(
        title=dict(
            text="Precision-Recall Curve",
            font=dict(
                color="#22D3EE",
                size=20,
            ),
        ),
        xaxis=dict(
            title="Recall",
            color="#CBD5E1",
            gridcolor="rgba(148,163,184,0.10)",
        ),
        yaxis=dict(
            title="Precision",
            color="#CBD5E1",
            gridcolor="rgba(148,163,184,0.10)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(
                color="#CBD5E1",
            ),
        ),
    )

    st.plotly_chart(
        fig_pr,
        width="stretch",
    )


st.divider()


# ============================================================
# MODEL INSPECTION
# ============================================================

st.subheader("Inspect a Model")

selected_model = st.selectbox(
    "Model",
    model_names,
    index=model_names.index(
        best_model_name
    ),
)

threshold = st.slider(
    "Classification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)


# ============================================================
# PREDICTION
# ============================================================

proba = proba_map[selected_model]

y_pred = (
    proba >= threshold
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)


c3, c4 = st.columns(2)


with c3:

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=[
            [0.0, "#0B1020"],
            [0.25, "#1E1B4B"],
            [0.5, "#4338CA"],
            [0.75, "#8B5CF6"],
            [1.0, "#22D3EE"],
        ],
        x=[
            "Pred 0",
            "Pred 1",
        ],
        y=[
            "True 0",
            "True 1",
        ],
        title=(
            f"{selected_model} — "
            f"Confusion Matrix "
            f"(threshold={threshold:.2f})"
        ),
    )

    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#CBD5E1",
        ),
        title=dict(
            font=dict(
                color="#A78BFA",
                size=20,
            ),
        ),
    )

    st.plotly_chart(
        fig_cm,
        width="stretch",
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

with c4:

    if selected_model in importances:

        imp = importances[
            selected_model
        ]

        label = (
            "Coefficient"
            if selected_model
            == "Logistic Regression"
            else "Importance"
        )

        fig_imp = px.bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            labels={
                "x": label,
                "y": "Feature",
            },
            title=(
                f"{selected_model} — "
                f"Feature {label}"
            ),
        )

        fig_imp.update_traces(
            marker=dict(
                color="#8B5CF6",
            ),
        )

        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#CBD5E1",
            ),
            title=dict(
                font=dict(
                    color="#22D3EE",
                    size=20,
                ),
            ),
            xaxis=dict(
                gridcolor=(
                    "rgba(148,163,184,0.10)"
                ),
                color="#CBD5E1",
            ),
            yaxis=dict(
                gridcolor=(
                    "rgba(148,163,184,0.10)"
                ),
                color="#CBD5E1",
            ),
        )

        st.plotly_chart(
            fig_imp,
            width="stretch",
        )

    else:

        st.info(
            f"{selected_model} "
            "does not expose feature importances."
        )


st.divider()


# ============================================================
# REAL TEST SET PREDICTIONS
# ============================================================

st.subheader(
    "Predictions on the Real Test Set"
)

if submission is not None:

    st.caption(
        f"Scored with the best model "
        f"({best_model_name}). "
        f"{len(submission):,} rows."
    )

    default_rate = (
        submission["predicted_class"].mean()
    )

    prediction_col1, prediction_col2 = st.columns(2)

    prediction_col1.metric(
        "Predicted Default Rate",
        f"{default_rate:.2%}",
    )

    prediction_col2.metric(
        "Total Predictions",
        f"{len(submission):,}",
    )

    st.dataframe(
        submission.head(200),
        width="stretch",
    )

    st.download_button(
        label="Download Full Predictions (CSV)",
        data=submission.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )

else:

    st.info(
        "No test-set predictions found. "
        "Run the training pipeline to "
        "generate them."
    )