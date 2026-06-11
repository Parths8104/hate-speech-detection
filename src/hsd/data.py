"""Dataset loading and splitting.

Expects a CSV with a text column and a label column (configurable). Works with
the public Davidson et al. (2017) "Hate Speech and Offensive Language" dataset
out of the box; see data/README.md for how to fetch it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from hsd.config import Config


@dataclass
class Dataset:
    X_train: list[str]
    X_test: list[str]
    y_train: list
    y_test: list
    labels: list[str]


def load_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data.path)
    missing = {cfg.data.text_column, cfg.data.label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df = df[[cfg.data.text_column, cfg.data.label_column]].dropna()
    df = df.rename(
        columns={cfg.data.text_column: "text", cfg.data.label_column: "label"}
    )
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    if cfg.data.sample_size:
        df = df.sample(
            n=min(cfg.data.sample_size, len(df)), random_state=cfg.seed
        ).reset_index(drop=True)
    return df


def build_dataset(cfg: Config) -> Dataset:
    df = load_dataframe(cfg)
    labels = sorted(df["label"].unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
        stratify=df["label"].tolist(),
    )
    return Dataset(X_train, X_test, y_train, y_test, labels)
