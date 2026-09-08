from types import SimpleNamespace

from regression.detector import detect_regression


def make_result(
    factuality,
    relevance,
    format_compliance,
    faithfulness,
    overall_score,
):
    return SimpleNamespace(
        factuality=factuality,
        relevance=relevance,
        format_compliance=format_compliance,
        faithfulness=faithfulness,
        overall_score=overall_score,
    )


def test_no_regression():

    before = make_result(
        0.90,
        0.90,
        0.90,
        0.90,
        0.90,
    )

    after = make_result(
        0.88,
        0.89,
        0.91,
        0.89,
        0.89,
    )

    result = detect_regression(before, after)

    assert result.regression_detected is False


def test_overall_regression():

    before = make_result(
        0.90,
        0.90,
        0.90,
        0.90,
        0.90,
    )

    after = make_result(
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
    )

    result = detect_regression(before, after)

    assert result.overall_regression is True
    assert result.regression_detected is True


def test_metric_regression():

    before = make_result(
        0.90,
        0.90,
        0.90,
        0.90,
        0.90,
    )

    after = make_result(
        0.90,
        0.90,
        0.90,
        0.79,
        0.87,
    )

    result = detect_regression(before, after)

    assert result.overall_regression is False
    assert result.metric_regression is True
    assert result.regression_detected is True


def test_exact_metric_threshold():

    before = make_result(
        0.90,
        0.90,
        0.90,
        0.90,
        0.90,
    )

    after = make_result(
        0.90,
        0.90,
        0.90,
        0.80,
        0.88,
    )

    result = detect_regression(before, after)

    assert result.metric_regression is True