from evaluation.models import EvaluationResult


WEIGHTS = {
    "factuality": 0.30,
    "relevance": 0.25,
    "format_compliance": 0.20,
    "faithfulness": 0.25,
}


def calculate_overall_score(
    result: EvaluationResult,
) -> float:

    score = (
        result.factuality * WEIGHTS["factuality"]
        + result.relevance * WEIGHTS["relevance"]
        + result.format_compliance * WEIGHTS["format_compliance"]
        + result.faithfulness * WEIGHTS["faithfulness"]
    )

    return round(score, 4)