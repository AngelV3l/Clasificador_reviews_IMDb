import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from sentiment_classifier.train import build_pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API backed by a tiny model trained on toy data, writing to a temp dir."""
    model_path = tmp_path / "sentiment_pipeline.joblib"
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    texts = ["great wonderful film", "terrible awful boring movie"] * 20
    labels = ["positive", "negative"] * 20
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.chdir(tmp_path)  # keep the predictions DB out of the repo

    from sentiment_classifier import api
    from sentiment_classifier.model import load_pipeline

    load_pipeline.cache_clear()
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_label_and_confidence(client):
    resp = client.post("/predict", json={"review_text": "a wonderful great film"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"positive", "negative"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"review_text": ""}).status_code == 422
