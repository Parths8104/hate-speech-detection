import numpy as np
import pytest

from hsd.evaluate import compute_pr_auc


def test_compute_pr_auc_perfect_scores():
    y_true = ["hate", "offensive", "neither", "hate", "offensive", "neither"]
    labels = ["hate", "neither", "offensive"]

    y_score = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.90, 0.05],
            [0.85, 0.10, 0.05],
            [0.05, 0.10, 0.85],
            [0.05, 0.85, 0.10],
        ]
    )

    metrics = compute_pr_auc(y_true, y_score, labels)

    assert metrics["pr_auc_macro"] == 1.0
    assert metrics["pr_auc_hate"] == 1.0
    assert metrics["pr_auc_neither"] == 1.0
    assert metrics["pr_auc_offensive"] == 1.0


def test_compute_pr_auc_rejects_wrong_score_shape():
    y_true = ["hate", "offensive", "neither"]
    labels = ["hate", "neither", "offensive"]

    y_score = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(
        ValueError,
        match="y_score must have one probability column per label",
    ):
        compute_pr_auc(y_true, y_score, labels)
