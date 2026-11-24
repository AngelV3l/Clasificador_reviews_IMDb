# IMDb Sentiment Classifier

Classifies English movie reviews as **positive** or **negative**. A TF-IDF +
logistic-regression pipeline trained on the 50k-review IMDb dataset, served two
ways: a FastAPI JSON endpoint and a Gradio web demo.

[![Tests](https://github.com/AngelV3l/Clasificador_reviews_IMDb/actions/workflows/tests.yml/badge.svg)](https://github.com/AngelV3l/Clasificador_reviews_IMDb/actions/workflows/tests.yml)

> **Deployment status:** deployment-ready but not yet hosted. The app runs locally
> (`python app.py`) and ships with a `Dockerfile`; see [Deployment](#deployment)
> for the free hosting options being evaluated.

## Results

Five-fold CV on the training split, plus a held-out test set (deduplicated, 20% test):

| Model                | CV accuracy       | Test accuracy | Test F1 (macro) |
| -------------------- | ----------------- | ------------- | --------------- |
| Logistic Regression* | 0.894 ± 0.002     | **0.896**     | **0.896**       |
| Linear SVM           | 0.889 ± 0.001     | 0.893         | 0.893           |
| Naive Bayes          | 0.866 ± 0.002     | 0.870         | 0.870           |

\* Served model. Logistic regression and the linear SVM are effectively tied, but
logistic regression exposes calibrated-ish probabilities, which the demo shows as
a confidence score.

## Design decisions

- **Cleaning lives inside the pipeline.** `TfidfVectorizer(preprocessor=clean_text)`
  means the exact same normalisation runs at training and at inference. An earlier
  version cleaned text only for training and fed raw text at serve time — a
  train/serve skew that silently degraded predictions.
- **Duplicates dropped before splitting.** The raw dataset has 418 duplicate
  reviews; removing them before `train_test_split` avoids leakage between train
  and test.
- **One artifact.** The vectoriser and classifier are pickled together as a single
  `Pipeline`, so they can never fall out of sync.
- **Model choice driven by the product.** Accuracy alone was a tie; the deciding
  factor was `predict_proba` for the UI.

## Project layout

```
src/sentiment_classifier/
    preprocessing.py   # clean_text, shared by train and serve
    train.py           # dedupe -> split -> compare 3 models -> save pipeline
    model.py           # cached loader + predict()
    api.py             # FastAPI service (/health, /predict)
    gradio_app.py      # Gradio UI
    storage.py         # SQLite prediction log
app.py                 # Gradio entry point (local + Hugging Face Spaces)
notebooks/             # 01_eda, 02_modeling
tests/                 # pytest (preprocessing + API)
```

## Quickstart

```bash
pip install -e ".[dev]"

# Retrain from the raw CSV (see "Data" below); regenerates models/
python -m sentiment_classifier.train

# Web demo
python app.py

# REST API
uvicorn sentiment_classifier.api:app --reload
# curl -X POST localhost:8000/predict -H "content-type: application/json" \
#      -d '{"review_text": "A gripping, brilliant film"}'

pytest
```

## Data

The raw dataset (`data/IMDB_reviews.csv`, ~66 MB, columns `review,sentiment`) is
not versioned. Download the *IMDB Dataset of 50K Movie Reviews* from
[Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
and place it there. The trained pipeline is committed, so the demo runs without
this step.

## Deployment

Not deployed yet. The pieces are in place — `app.py` as the entry point, a
`Dockerfile`, and a committed model so no training step is needed at deploy time.

Free options under consideration:

- **Streamlit Community Cloud** — free for public apps, deploys straight from the
  GitHub repo (would require a small Streamlit UI alongside the Gradio one).
- **Render (free tier)** — runs the current Gradio/FastAPI stack from the
  `Dockerfile`; the instance sleeps after inactivity and cold-starts on the next
  request.
