from evaluation.models import EvaluationResult
from evaluation.judge import judge_response
from evaluation.deterministic import evaluate_deterministic


def evaluate_response(
    test_case,
    generated_response,
    judge_provider,
) -> EvaluationResult:

    if test_case.evaluation_type == "llm_judge":

        result = judge_response(
            test_case=test_case,
            response_text=generated_response.response,
            provider=judge_provider,
        )

    elif test_case.evaluation_type == "deterministic":

        result = evaluate_deterministic(
            test_case=test_case,
            response_text=generated_response.response,
        )

        result["judge_model"] = "deterministic"

    else:
        raise ValueError(
            f"Unsupported evaluation type: "
            f"{test_case.evaluation_type}"
        )

    return EvaluationResult(
        test_id=test_case.id,
        factuality=result["factuality"],
        relevance=result["relevance"],
        format_compliance=result["format_compliance"],
        faithfulness=result["faithfulness"],
        reason=result["reason"],
        judge_model=result["judge_model"],
    )