"""Text normalization implemented as a scikit-learn transformer.

Keeping cleaning inside the pipeline (rather than as a one-off preprocessing
step) means the exact same transform is applied at train and inference time,
which removes a whole class of training/serving skew bugs.
"""
from __future__ import annotations

import re

from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 2
WS_RE = re.compile(r"\s+")
# Collapse the most common HTML entities seen in scraped social data.
ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|#\d+);")


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    strip_hashtags: bool = False,
    collapse_repeats: bool = True,
) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = ENTITY_RE.sub(" ", text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_mentions:
        text = MENTION_RE.sub(" ", text)
    # Hashtags: either drop entirely, or keep the word and drop the '#'.
    text = HASHTAG_RE.sub(" " if strip_hashtags else r" \1 ", text)
    if collapse_repeats:
        text = REPEAT_RE.sub(r"\1\1", text)
    if lowercase:
        text = text.lower()
    text = WS_RE.sub(" ", text).strip()
    return text


class TextCleaner(BaseEstimator, TransformerMixin):
    """Vectorized text cleaner usable as the first step of a Pipeline."""

    def __init__(
        self,
        lowercase: bool = True,
        strip_urls: bool = True,
        strip_mentions: bool = True,
        strip_hashtags: bool = False,
        collapse_repeats: bool = True,
    ):
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.strip_mentions = strip_mentions
        self.strip_hashtags = strip_hashtags
        self.collapse_repeats = collapse_repeats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            clean_text(
                x,
                lowercase=self.lowercase,
                strip_urls=self.strip_urls,
                strip_mentions=self.strip_mentions,
                strip_hashtags=self.strip_hashtags,
                collapse_repeats=self.collapse_repeats,
            )
            for x in X
        ]
