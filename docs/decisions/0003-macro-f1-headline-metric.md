# ADR 0003 — Use macro F1 as the headline metric

## Status
Accepted

## Context
The dataset is imbalanced across `hate`, `offensive`, and `neither`. Accuracy can look
strong even when the classifier performs poorly on a minority class, because correct
predictions on the majority class dominate the score.

For this system, performance across all three classes matters. A headline metric should
therefore expose weak minority-class performance instead of allowing it to be hidden by
class frequency.

## Decision
Use **macro F1** as the primary evaluation and cross-validation metric.

Macro F1 computes F1 independently for each class and then gives every class equal
weight, regardless of how frequently that class appears in the dataset.

Report it alongside a **majority-class baseline**, per-class precision/recall/F1,
weighted F1, accuracy, and the confusion matrix. The additional metrics remain useful
for diagnosis, but macro F1 is the headline number used when comparing model changes.

## Consequences
- Minority-class performance contributes equally to the headline metric.
- Model improvements cannot rely only on getting the majority class right.
- Accuracy and weighted F1 remain available for context but are not used as the primary
  measure of model quality.
- Macro F1 can fall sharply when one class performs poorly, which is intentional because
  it makes that failure visible.
- If deployment requirements later prioritize a specific class or operating threshold,
  metrics such as per-class recall or PR-AUC may become more important and this decision
  should be revisited.
