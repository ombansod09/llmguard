import sqlite3
from pathlib import Path
from uuid import uuid4


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
            run_id TEXT NOT NULL,
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


def create_run_id() -> str:
    return str(uuid4())


def save_evaluation(
    run_id: str,
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
            run_id,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
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
    run_id: str | None = None,
    prompt_version: str | None = None,
):
    connection = get_connection()

    if run_id:
        cursor = connection.execute(
            """
            SELECT *
            FROM experiments
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        )

    elif prompt_version:
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


def get_run_ids():
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT
            run_id,
            dataset_version,
            prompt_version,
            requested_model,
            MIN(created_at) AS created_at,
            COUNT(*) AS test_count
        FROM experiments
        GROUP BY run_id
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows