"""Convert the Davidson et al. (2017) raw CSV into this project's (text, label) format.

Usage:
    python scripts/prepare_davidson.py --in labeled_data.csv --out data/dataset.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

CLASS_MAP = {0: "hate", 1: "offensive", 2: "neither"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Davidson dataset.")
    parser.add_argument("--in", dest="inp", required=True, help="Path to labeled_data.csv")
    parser.add_argument("--out", dest="out", default="data/dataset.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    if "tweet" not in df.columns or "class" not in df.columns:
        raise SystemExit(
            "Expected columns 'tweet' and 'class' in the Davidson CSV. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {"text": df["tweet"].astype(str), "label": df["class"].map(CLASS_MAP)}
    ).dropna()
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
