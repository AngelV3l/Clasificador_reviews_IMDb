import pytest

from sentiment_classifier.gradio_app import format_confidence


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.5, "50.0%"),
        (0.9734, "97.3%"),
        (0.9984, "99.8%"),
        (0.999, "> 99.9%"),
        (0.99996, "> 99.9%"),
    ],
)
def test_format_confidence_never_shows_a_flat_100(confidence, expected):
    assert format_confidence(confidence) == expected
