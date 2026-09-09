import sqlite3

from storage import database


def test_database_initialization(tmp_path, monkeypatch):

    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DB_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        database,
        "DB_PATH",
        db_path,
    )

    database.initialize_database()

    connection = sqlite3.connect(db_path)

    cursor = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'experiments'
        """
    )

    table = cursor.fetchone()

    connection.close()

    assert table is not None


def test_save_and_retrieve(tmp_path, monkeypatch):

    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DB_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        database,
        "DB_PATH",
        db_path,
    )

    database.initialize_database()

    database.save_evaluation(
        run_id="run_test_001",
        dataset_version="benchmark_v1",
        prompt_version="v1",
        requested_model="openrouter/free",
        actual_model="test-model",
        test_id="fact_001",
        response="Paris.",
        factuality=1.0,
        relevance=1.0,
        format_compliance=1.0,
        faithfulness=1.0,
        overall_score=1.0,
        reason="Correct answer.",
        judge_model="test-judge",
    )

    rows = database.get_experiment_results(
        run_id="run_test_001"
    )

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_test_001"
    assert rows[0]["test_id"] == "fact_001"
    assert rows[0]["overall_score"] == 1.0