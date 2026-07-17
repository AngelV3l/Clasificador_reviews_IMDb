"""FastAPI service around the sentiment pipeline."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sentiment_classifier.model import load_pipeline, predict
from sentiment_classifier.storage import log_prediction

logger = logging.getLogger("sentiment_classifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()  # fail fast and warm the cache at startup
    logger.info("Model loaded.")
    yield


app = FastAPI(title="IMDb Sentiment Classifier", version="0.1.0", lifespan=lifespan)


class ReviewIn(BaseModel):
    review_text: str = Field(..., min_length=1, examples=["A masterful, gripping film."])


class PredictionOut(BaseModel):
    label: str
    confidence: float | None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(payload: ReviewIn) -> PredictionOut:
    label, confidence = predict(payload.review_text)
    log_prediction(payload.review_text, label, confidence)
    return PredictionOut(label=label, confidence=confidence)
