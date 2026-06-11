"""Evaluation harness: metrics, per-class report, and a confusion matrix plot.

Macro F1 is the headline metric rather than accuracy, because hate speech
datasets are heavily imbalanced and accuracy rewards a model that simply
predicts the majority class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def per_class_report(y_true, y_pred, labels: list[str]) -> dict:
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )


def save_confusion_matrix(y_true, y_pred, labels: list[str], out_path: str | Path) -> None:
    # Imported lazily so the core library does not hard-depend on matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "#15201b",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def baseline_majority_f1(y_train, y_test) -> float:
    """Macro F1 of a majority-class baseline, for an honest lift comparison."""
    from collections import Counter

    majority = Counter(y_train).most_common(1)[0][0]
    y_pred = [majority] * len(y_test)
    return round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
