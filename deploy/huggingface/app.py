"""Hugging Face Spaces entry point.

Free Spaces run on ZeroGPU hardware, which refuses to start unless at least one
function is decorated with ``@spaces.GPU``. This model is a TF-IDF + logistic
regression pipeline that runs on CPU, so predictions never request a GPU: the
decorated function below exists only to pass that startup check.
"""

import sys
from pathlib import Path

import spaces

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentiment_classifier.gradio_app import demo


@spaces.GPU
def _zerogpu_startup_check() -> None:
    """Never called. ZeroGPU only needs a decorated function to exist."""


if __name__ == "__main__":
    demo.launch()
