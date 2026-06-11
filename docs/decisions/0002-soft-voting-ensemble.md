# ADR 0002 — Calibrated soft-voting ensemble

## Status
Accepted

## Context
Linear models (logistic regression, linear SVC) are strong on sparse
high-dimensional TF-IDF features. Tree ensembles capture different, non-linear
interactions. No single model dominates across all three classes, and the minority
`hate` class benefits from combining decision boundaries.

## Decision
A `VotingClassifier` with **soft** voting over logistic regression, a calibrated
linear SVC, and a random forest. LinearSVC has no native `predict_proba`, so it is
wrapped in `CalibratedClassifierCV` to produce calibrated probabilities suitable for
soft voting and for downstream confidence scores in the API. Linear models are
weighted higher than the forest, reflecting their edge on sparse text features.

## Consequences
- More stable predictions than any single estimator, especially on the minority class.
- Calibrated probabilities are meaningful as confidence scores at serving time.
- Higher training cost; acceptable for this dataset size and offline training.
