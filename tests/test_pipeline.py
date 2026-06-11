"""End-to-end smoke test: the pipeline should fit on tiny data and predict a known label."""
from hsd.config import Config
from hsd.models import build_pipeline


def _tiny_config() -> Config:
    return Config(
        seed=0,
        data={"path": "data/sample.csv"},
        features={
            "word": {"ngram_range": (1, 1), "min_df": 1, "max_features": 1000},
            "char": {"ngram_range": (2, 3), "min_df": 1, "max_features": 1000},
        },
        model={
            "weights": [2, 2, 1],
            "random_forest": {"n_estimators": 20, "n_jobs": 1},
        },
    )


def test_pipeline_fits_and_predicts():
    texts = [
        "what a lovely sunny day at the park",
        "the food was delicious and the staff were kind",
        "you are an absolute idiot and a fool",
        "shut up you clueless moron",
        "great work on the project everyone",
        "stop being such a clown all the time",
    ]
    labels = ["neither", "neither", "offensive", "offensive", "neither", "offensive"]

    pipe = build_pipeline(_tiny_config())
    pipe.fit(texts, labels)

    pred = pipe.predict(["you are a complete idiot"])[0]
    assert pred in {"neither", "offensive"}

    # Soft-voting ensemble must expose probabilities for the API/predict layer.
    proba = pipe.predict_proba(["lovely day at the park"])[0]
    assert abs(sum(proba) - 1.0) < 1e-6
