"""Evaluate a fitted classifier and report/plot its metrics."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name, model, X_te, y_te, threshold=0.5, verbose=True):
    """Compute standard classification metrics and optionally print/plot them.

    Returns (metrics_dict, y_proba).
    """
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_proba),
    }

    if verbose:
        print(f"===== {name} — Quick Evaluation =====")
        for k, v in metrics.items():
            if k != "model":
                print(f"{k:>10}: {v:.4f}")
        print("\nClassification report:")
        print(classification_report(y_te, y_pred, digits=3, zero_division=0))

        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
        )
        ax.set_title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return metrics, y_proba
