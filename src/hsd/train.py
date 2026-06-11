"""Train the model: cross-validate, fit, evaluate on a held-out split, and save artifacts.

Run:
    python -m hsd.train --config config/config.yaml
or, after `pip install -e .`:
    hsd-train --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from hsd.config import Config, load_config
from hsd.data import build_dataset
from hsd.evaluate import (
    baseline_majority_f1,
    compute_metrics,
    per_class_report,
    save_confusion_matrix,
)
from hsd.models import build_pipeline


def _maybe_grid_search(pipeline, X_train, y_train, cfg: Config):
    """Light grid over the linear models' regularization strength."""
    grid = {
        "clf__logreg__C": [0.5, 1.0, 2.0],
        "clf__svc__estimator__C": [0.5, 1.0, 2.0],
    }
    cv = StratifiedKFold(n_splits=cfg.train.cv_folds, shuffle=True, random_state=cfg.seed)
    search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"[grid] best params: {search.best_params_}")
    print(f"[grid] best CV f1_macro: {search.best_score_:.4f}")
    return search.best_estimator_


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hate speech detector.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)

    print("[1/5] Loading data...")
    ds = build_dataset(cfg)
    print(f"      train={len(ds.X_train)}  test={len(ds.X_test)}  classes={ds.labels}")

    pipeline = build_pipeline(cfg)

    print("[2/5] Cross-validating...")
    cv = StratifiedKFold(n_splits=cfg.train.cv_folds, shuffle=True, random_state=cfg.seed)
    cv_scores = cross_val_score(
        pipeline, ds.X_train, ds.y_train, scoring="f1_macro", cv=cv, n_jobs=-1
    )
    print(f"      CV f1_macro: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("[3/5] Fitting final model...")
    start = time.time()
    if cfg.train.grid_search:
        pipeline = _maybe_grid_search(pipeline, ds.X_train, ds.y_train, cfg)
    else:
        pipeline.fit(ds.X_train, ds.y_train)
    fit_seconds = round(time.time() - start, 1)

    print("[4/5] Evaluating on held-out test set...")
    y_pred = pipeline.predict(ds.X_test)
    metrics = compute_metrics(ds.y_test, y_pred)
    baseline_f1 = baseline_majority_f1(ds.y_train, ds.y_test)
    metrics["cv_f1_macro_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_f1_macro_std"] = round(float(cv_scores.std()), 4)
    metrics["baseline_majority_f1_macro"] = baseline_f1
    metrics["lift_over_baseline_f1_macro"] = round(metrics["f1_macro"] - baseline_f1, 4)
    metrics["fit_seconds"] = fit_seconds
    for k, v in metrics.items():
        print(f"      {k:32s} {v}")

    print("[5/5] Saving artifacts...")
    artifact_dir = Path(cfg.train.artifact_dir)
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipeline": pipeline, "labels": ds.labels}, artifact_dir / "model.joblib")
    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(reports_dir / "classification_report.json", "w") as f:
        json.dump(per_class_report(ds.y_test, y_pred, ds.labels), f, indent=2)
    save_confusion_matrix(ds.y_test, y_pred, ds.labels, reports_dir / "confusion_matrix.png")

    print(f"\nDone. Model and metrics written to {artifact_dir}/")


if __name__ == "__main__":
    main()
