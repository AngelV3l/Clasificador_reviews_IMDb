"""Load the trained pipeline and expose a single prediction entry point."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def default_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "models/sentiment_pipeline.joblib"))


@lru_cache(maxsize=1)
def load_pipeline(model_path: Path | None = None) -> Pipeline:
    path = model_path or default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Run `python -m sentiment_classifier.train` first."
        )
    return joblib.load(path)


def predict(text: str) -> tuple[str, float | None]:
    """Return (label, confidence). Confidence is None if the model has no probabilities."""
    pipeline = load_pipeline()
    label = str(pipeline.predict([text])[0])
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        confidence = float(proba.max())
    return label, confidence
