"""FastAPI inference service.

Run:
    uvicorn hsd.api:app --reload
Then POST to /predict, or open /docs for the interactive UI.
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hsd.predict import DEFAULT_MODEL, Predictor

app = FastAPI(
    title="Hate Speech Detection API",
    description="Classifies text and returns calibrated per-class scores.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    confidence: float | None = None
    scores: dict[str, float] | None = None


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL)
    return Predictor(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train one first (python -m hsd.train).",
        )
    result = predictor.predict(req.text)
    return PredictResponse(
        label=result["label"],
        confidence=result.get("confidence"),
        scores=result.get("scores"),
    )
