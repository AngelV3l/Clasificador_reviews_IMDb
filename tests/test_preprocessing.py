from sentiment_classifier.preprocessing import clean_text


def test_lowercases_and_strips_html():
    assert clean_text("<br/>GREAT Movie!") == " great movie "


def test_removes_digits_and_punctuation():
    assert clean_text("Rated 10/10!!!").split() == ["rated"]


def test_idempotent_on_clean_input():
    cleaned = clean_text("a simple sentence")
    assert clean_text(cleaned) == cleaned
