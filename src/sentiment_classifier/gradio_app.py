"""Gradio UI for the sentiment classifier (shared by local run and HF Spaces)."""

import gradio as gr

from sentiment_classifier.model import predict
from sentiment_classifier.storage import log_prediction

EXAMPLES = [
    "One of the best films I have seen this year. The performances were stunning.",
    "A complete waste of time. Predictable plot and wooden acting.",
]


def format_confidence(confidence: float) -> str:
    """Show one decimal, and never round a probability up to a flat 100%."""
    if confidence >= 0.999:
        return "> 99.9%"
    return f"{confidence:.1%}"


def classify(review: str) -> str:
    review = review.strip()
    if not review:
        return "Please enter a review."
    label, confidence = predict(review)
    log_prediction(review, label, confidence)
    if confidence is None:
        return f"Sentiment: {label}"
    return f"Sentiment: {label} ({format_confidence(confidence)} confidence)"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=6, label="Movie review", placeholder="Write a review in English..."),
    outputs=gr.Textbox(label="Prediction"),
    title="IMDb Sentiment Classifier",
    description="TF-IDF + logistic regression trained on the 50k-review IMDb dataset (89.6% test accuracy).",
    examples=EXAMPLES,
    flagging_mode="never",
)
