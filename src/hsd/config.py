"""Typed configuration loaded from a YAML file.

Using pydantic gives validation and editor autocomplete, and keeps every
tunable in one versioned place rather than scattered through the code.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataCfg(BaseModel):
    path: str
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.2
    sample_size: int | None = None


class PreprocessCfg(BaseModel):
    lowercase: bool = True
    strip_urls: bool = True
    strip_mentions: bool = True
    strip_hashtags: bool = False
    collapse_repeats: bool = True
    min_token_len: int = 1


class VectorizerCfg(BaseModel):
    ngram_range: tuple[int, int]
    min_df: int = 3
    max_features: int = 50000
    sublinear_tf: bool = True


class FeaturesCfg(BaseModel):
    word: VectorizerCfg
    char: VectorizerCfg


class LogRegCfg(BaseModel):
    C: float = 1.0
    max_iter: int = 2000


class LinearSVCCfg(BaseModel):
    C: float = 1.0


class RandomForestCfg(BaseModel):
    n_estimators: int = 300
    max_depth: int | None = None
    n_jobs: int = -1


class ModelCfg(BaseModel):
    weights: list[float] = Field(default_factory=lambda: [2, 2, 1])
    logreg: LogRegCfg = LogRegCfg()
    linear_svc: LinearSVCCfg = LinearSVCCfg()
    random_forest: RandomForestCfg = RandomForestCfg()


class TrainCfg(BaseModel):
    cv_folds: int = 5
    grid_search: bool = False
    artifact_dir: str = "models"


class Config(BaseModel):
    seed: int = 42
    data: DataCfg
    preprocess: PreprocessCfg = PreprocessCfg()
    features: FeaturesCfg
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
