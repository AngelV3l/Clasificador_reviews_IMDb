"""Entry point for the Gradio demo (also the Hugging Face Spaces entry point).

Runs without installing the package first, so it works both locally
(`python app.py`) and on Spaces, which only installs requirements.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentiment_classifier.gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch()
