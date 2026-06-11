"""Load a trained model and classify text.

Run:
    python -m hsd.predict "some text to classify"
    python -m hsd.predict            # interactive mode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

DEFAULT_MODEL = "models/model.joblib"


class Predictor:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.labels = bundle["labels"]

    def predict(self, text: str) -> dict:
        label = self.pipeline.predict([text])[0]
        result = {"text": text, "label": label}
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba([text])[0]
            classes = list(self.pipeline.classes_)
            result["scores"] = {
                c: round(float(p), 4) for c, p in zip(classes, proba)
            }
            result["confidence"] = round(float(max(proba)), 4)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify text with a trained model.")
    parser.add_argument("text", nargs="*", help="Text to classify. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"Model not found at {args.model}. Train one first with: python -m hsd.train")

    predictor = Predictor(args.model)

    if args.text:
        result = predictor.predict(" ".join(args.text))
        _print(result)
        return

    print("Interactive mode. Type text to classify, or 'q' to quit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"q", "quit", "exit"}:
            break
        if line:
            _print(predictor.predict(line))


def _print(result: dict) -> None:
    print(f"  label      : {result['label']}")
    if "confidence" in result:
        print(f"  confidence : {result['confidence']}")
        print("  scores     :")
        for c, p in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
            print(f"    {c:14s} {p}")


if __name__ == "__main__":
    main()
