import sqlite3
from pathlib import Path


DB_DIR = Path("storage")
DB_PATH = DB_DIR / "llmguard.db"


def get_connection():
    DB_DIR.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_version TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            requested_model TEXT NOT NULL,
            actual_model TEXT NOT NULL,
            test_id TEXT NOT NULL,
            response TEXT NOT NULL,
            factuality REAL NOT NULL,
            relevance REAL NOT NULL,
            format_compliance REAL NOT NULL,
            faithfulness REAL NOT NULL,
            overall_score REAL NOT NULL,
            reason TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()

def save_evaluation(
    dataset_version: str,
    prompt_version: str,
    requested_model: str,
    actual_model: str,
    test_id: str,
    response: str,
    factuality: float,
    relevance: float,
    format_compliance: float,
    faithfulness: float,
    overall_score: float,
    reason: str,
    judge_model: str,
):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO experiments (
            dataset_version,
            prompt_version,
            requested_model,
            actual_model,
            test_id,
            response,
            factuality,
            relevance,
            format_compliance,
            faithfulness,
            overall_score,
            reason,
            judge_model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_version,
            prompt_version,
            requested_model,
            actual_model,
            test_id,
            response,
            factuality,
            relevance,
            format_compliance,
            faithfulness,
            overall_score,
            reason,
            judge_model,
        ),
    )

    connection.commit()
    connection.close()

def get_experiment_results(
    prompt_version: str | None = None,
):
    connection = get_connection()

    if prompt_version:
        cursor = connection.execute(
            """
            SELECT *
            FROM experiments
            WHERE prompt_version = ?
            ORDER BY id
            """,
            (prompt_version,),
        )
    else:
        cursor = connection.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY id
            """
        )

    rows = cursor.fetchall()

    connection.close()

    return rows