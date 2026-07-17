"""Text normalisation shared by training and inference.

`clean_text` is wired into the TF-IDF vectoriser as its ``preprocessor``, so the
exact same normalisation runs at train time and at serving time. Keeping it here
(importable by name) is what lets the pickled pipeline round-trip cleanly.
"""

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_LETTER = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, and drop non-alphabetic characters."""
    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    return text
