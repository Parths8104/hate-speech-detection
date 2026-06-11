# Model Card — Hate Speech Detection

## Overview

A text classifier that labels short documents as `hate`, `offensive`, or `neither`.
It combines word and character TF-IDF features with a calibrated soft-voting
ensemble (logistic regression, linear SVC, random forest).

## Intended use

- **Intended:** as a first-pass triage signal in content-moderation tooling, to
  surface likely-harmful content for human review, and as a teaching example of a
  reproducible classical-ML text-classification pipeline.
- **Not intended:** as a fully automated moderation decision-maker. Predictions
  should support human reviewers, not replace them. The model should not be used
  to take punitive action against users without human oversight.

## Training data

Designed for the public Davidson et al. (2017) dataset (~24.8k labeled tweets,
three classes). The dataset is imbalanced: the `offensive` class dominates, `hate`
is the minority class. See `data/README.md`. No dataset is redistributed in this
repo; only a tiny synthetic sample is committed.

## Evaluation

- Headline metric is **macro F1**, not accuracy, because the data is imbalanced and
  accuracy over-rewards predicting the majority class.
- Reported alongside a **majority-class baseline** so the lift is honest.
- 5-fold stratified cross-validation on the training split, plus a held-out test set.
- A row-normalized confusion matrix is saved to `models/reports/`.

Run `python -m hsd.train` to reproduce the numbers; results are written to
`models/metrics.json`. Report the numbers your run produces. Do not copy metrics
you have not reproduced.

## Limitations and risks

- **Class confusion:** the hardest boundary is `hate` vs `offensive`; profanity
  is not the same as targeted hate, and the model can conflate them. This is a
  known difficulty on this dataset, not a bug to paper over.
- **Annotation bias:** labels reflect the original annotators' judgments, which
  carry cultural and dialectal bias. Models trained on this data can over-flag
  African American English; this is documented in the dataset's own literature and
  must be weighed before any deployment.
- **Domain shift:** trained on tweets; performance will drop on long-form text,
  other platforms, or other time periods.
- **Adversarial evasion:** character n-grams help against light obfuscation, but a
  motivated adversary can still evade a static model.
- **Language:** word features are language-specific. Character n-grams generalize
  better, but the model should be retrained per target language or corpus.

## Ethical considerations

Hate speech classification carries real risk of unfair, biased outcomes if
deployed naively. This model is a triage aid, not an arbiter. Any deployment
should include human review, an appeals path, monitoring for disparate error
rates across groups, and periodic re-evaluation.
