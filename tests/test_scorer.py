from evaluation.models import EvaluationResult
from evaluation.scorer import calculate_overall_score


def test_weighted_score():
    result = EvaluationResult(
        test_id="test_001",
        factuality=1.0,
        relevance=0.8,
        format_compliance=0.9,
        faithfulness=0.7,
        reason="Test result",
        judge_model="test-model",
    )

    score = calculate_overall_score(result)

    expected = (
        1.0 * 0.30
        + 0.8 * 0.25
        + 0.9 * 0.20
        + 0.7 * 0.25
    )

    assert score == round(expected, 4)


def test_perfect_score():
    result = EvaluationResult(
        test_id="test_002",
        factuality=1.0,
        relevance=1.0,
        format_compliance=1.0,
        faithfulness=1.0,
        reason="Perfect",
        judge_model="test-model",
    )

    assert calculate_overall_score(result) == 1.0


def test_zero_score():
    result = EvaluationResult(
        test_id="test_003",
        factuality=0.0,
        relevance=0.0,
        format_compliance=0.0,
        faithfulness=0.0,
        reason="Zero",
        judge_model="test-model",
    )

    assert calculate_overall_score(result) == 0.0