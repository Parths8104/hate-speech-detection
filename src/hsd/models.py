"""Feature extraction and the ensemble pipeline.

Design notes:
- A FeatureUnion of word and character TF-IDF. Character n-grams are the key
  lever for abuse detection: they survive misspellings and deliberate
  obfuscation ("h@te", "stoopid") that defeat word-level features.
- A soft-voting ensemble of three complementary estimators. LinearSVC is wrapped
  in CalibratedClassifierCV so it exposes calibrated probabilities and can take
  part in soft voting alongside logistic regression and the random forest.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from hsd.config import Config
from hsd.preprocess import TextCleaner


def build_feature_union(cfg: Config) -> FeatureUnion:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg.features.word.ngram_range),
        min_df=cfg.features.word.min_df,
        max_features=cfg.features.word.max_features,
        sublinear_tf=cfg.features.word.sublinear_tf,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(cfg.features.char.ngram_range),
        min_df=cfg.features.char.min_df,
        max_features=cfg.features.char.max_features,
        sublinear_tf=cfg.features.char.sublinear_tf,
    )
    return FeatureUnion([("word", word), ("char", char)])


def build_estimator(cfg: Config) -> VotingClassifier:
    logreg = LogisticRegression(
        C=cfg.model.logreg.C,
        max_iter=cfg.model.logreg.max_iter,
        class_weight="balanced",
        random_state=cfg.seed,
    )
    # CalibratedClassifierCV gives LinearSVC predict_proba for soft voting.
    svc = CalibratedClassifierCV(
        LinearSVC(C=cfg.model.linear_svc.C, class_weight="balanced", random_state=cfg.seed),
        cv=3,
    )
    rf = RandomForestClassifier(
        n_estimators=cfg.model.random_forest.n_estimators,
        max_depth=cfg.model.random_forest.max_depth,
        class_weight="balanced_subsample",
        n_jobs=cfg.model.random_forest.n_jobs,
        random_state=cfg.seed,
    )
    return VotingClassifier(
        estimators=[("logreg", logreg), ("svc", svc), ("rf", rf)],
        voting="soft",
        weights=cfg.model.weights,
        n_jobs=cfg.model.random_forest.n_jobs,
    )


def build_pipeline(cfg: Config) -> Pipeline:
    return Pipeline(
        [
            (
                "clean",
                TextCleaner(
                    lowercase=cfg.preprocess.lowercase,
                    strip_urls=cfg.preprocess.strip_urls,
                    strip_mentions=cfg.preprocess.strip_mentions,
                    strip_hashtags=cfg.preprocess.strip_hashtags,
                    collapse_repeats=cfg.preprocess.collapse_repeats,
                ),
            ),
            ("features", build_feature_union(cfg)),
            ("clf", build_estimator(cfg)),
        ]
    )
