from types import SimpleNamespace

from regression.analyzer import find_regressions


def make_result(
    test_id,
    overall_score,
    reason,
):
    return SimpleNamespace(
        test_id=test_id,
        overall_score=overall_score,
        reason=reason,
    )


def test_find_regressions():

    before = [
        make_result(
            "fact_001",
            1.0,
            "Correct answer.",
        ),
        make_result(
            "format_001",
            1.0,
            "Correct format.",
        ),
    ]

    after = [
        make_result(
            "fact_001",
            0.98,
            "Correct answer.",
        ),
        make_result(
            "format_001",
            0.80,
            "Failed requested format.",
        ),
    ]

    regressions = find_regressions(
        before,
        after,
    )

    assert len(regressions) == 1

    assert regressions[0].test_id == "format_001"

    assert regressions[0].overall_before == 1.0
    assert regressions[0].overall_after == 0.80
    assert regressions[0].overall_change == -0.20

    assert regressions[0].reason_before == "Correct format."
    assert regressions[0].reason_after == "Failed requested format."


def test_regressions_are_sorted_by_worst_change():

    before = [
        make_result("test_001", 1.0, "Good."),
        make_result("test_002", 1.0, "Good."),
    ]

    after = [
        make_result("test_001", 0.80, "Worse."),
        make_result("test_002", 0.70, "Much worse."),
    ]

    regressions = find_regressions(
        before,
        after,
    )

    assert len(regressions) == 2

    assert regressions[0].test_id == "test_002"
    assert regressions[1].test_id == "test_001"


def test_ignore_small_change():

    before = [
        make_result(
            "test_001",
            1.0,
            "Good.",
        )
    ]

    after = [
        make_result(
            "test_001",
            0.96,
            "Slightly different.",
        )
    ]

    regressions = find_regressions(
        before,
        after,
    )

    assert regressions == []