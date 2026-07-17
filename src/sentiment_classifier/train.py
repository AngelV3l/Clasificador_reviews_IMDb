"""Train and evaluate the sentiment classifier.

Loads the raw IMDb CSV, removes duplicate reviews (they leak between the train
and test folds otherwise), compares three linear models with cross-validation,
and persists a single scikit-learn ``Pipeline`` plus a metrics report.

    python -m sentiment_classifier.train
    python -m sentiment_classifier.train --data data/IMDB_reviews.csv --out models/
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment_classifier.preprocessing import clean_text

RANDOM_STATE = 42


def build_pipeline(estimator) -> Pipeline:
    """TF-IDF (with in-pipeline cleaning) followed by a linear classifier."""
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=20000,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", estimator)])


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    before = len(df)
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    print(f"Loaded {before} reviews, {before - len(df)} duplicates removed, {len(df)} kept.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/IMDB_reviews.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=RANDOM_STATE,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "naive_bayes": MultinomialNB(),
        "linear_svc": LinearSVC(random_state=RANDOM_STATE),
    }

    report = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report[name] = {
            "cv_accuracy_mean": round(float(cv.mean()), 4),
            "cv_accuracy_std": round(float(cv.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }
        print(f"{name:>20}  cv={report[name]['cv_accuracy_mean']:.4f}  "
              f"test={report[name]['test_accuracy']:.4f}")

    # LinearSVC and LogisticRegression are statistically tied here, but only
    # LogisticRegression exposes predict_proba, which the demo uses for a
    # confidence score. That trade-off decides the served model.
    final = build_pipeline(candidates["logistic_regression"])
    final.fit(X_train, y_train)
    print("\nHeld-out classification report (logistic_regression):\n")
    print(classification_report(y_test, final.predict(X_test)))

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "sentiment_pipeline.joblib"
    joblib.dump(final, model_path)
    (args.out / "metrics.json").write_text(
        json.dumps({"served_model": "logistic_regression", "models": report}, indent=2)
    )
    print(f"\nSaved pipeline to {model_path}")


if __name__ == "__main__":
    main()
