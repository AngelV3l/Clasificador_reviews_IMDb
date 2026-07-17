"""SQLite logging of predictions.

Kept intentionally small: one table, parameterised inserts, and a path that
resolves relative to the current working directory unless overridden.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("predicciones.db")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def log_prediction(
    review_text: str,
    predicted_label: str,
    confidence: float | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (review_text, predicted_label, confidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            (review_text, predicted_label, confidence, datetime.now(timezone.utc).isoformat()),
        )
