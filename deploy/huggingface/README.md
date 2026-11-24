---
title: IMDb Sentiment Classifier
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: TF-IDF + logistic regression sentiment classifier for IMDb
---

# IMDb Sentiment Classifier

Classifies English movie reviews as positive or negative. A TF-IDF and
logistic regression pipeline trained on the 50k-review IMDb dataset
(0.896 accuracy on a held-out, deduplicated test set).

Source code, training pipeline, tests and FastAPI service:
[github.com/AngelV3l/Clasificador_reviews_IMDb](https://github.com/AngelV3l/Clasificador_reviews_IMDb)
