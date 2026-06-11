# ADR 0001 — Combine word and character n-grams

## Status
Accepted

## Context
Word-level TF-IDF is the obvious first choice for text classification, but abusive
text is adversarial. Users deliberately misspell and obfuscate ("h@te", "stoopid",
"id10t") and social text is noisy by default. Pure word features treat each variant
as an unseen token and lose the signal.

## Decision
Use a `FeatureUnion` of two TF-IDF vectorizers:
- **Word** n-grams (1-2) for normal lexical signal.
- **Character** n-grams (2-5, `char_wb`) to capture sub-word patterns that survive
  misspelling and obfuscation.

## Consequences
- Higher recall on obfuscated and misspelled abuse.
- Larger feature space and slower fitting; bounded with `max_features` and `min_df`.
- Character features are largely language-agnostic, which helps on code-mixed text.
