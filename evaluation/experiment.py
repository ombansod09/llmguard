from datasets.loader import load_dataset
from evaluation.evaluator import evaluate_response
from evaluation.scorer import calculate_overall_score
from llm.provider import OpenRouterProvider
from llm.runner import generate_response
from storage.database import create_run_id, save_evaluation


def run_experiment(
    dataset_version: str,
    prompt_version: str,
    temperature: float = 0.0,
    progress_callback=None,
):
    dataset = load_dataset(dataset_version)

    generation_provider = OpenRouterProvider()
    judge_provider = OpenRouterProvider()

    run_id = create_run_id()

    results = []
    failures = []

    total = len(dataset.tests)

    for index, test_case in enumerate(
        dataset.tests,
        start=1,
    ):

        try:
            generated = generate_response(
                test_case=test_case,
                prompt_version=prompt_version,
                provider=generation_provider,
                temperature=temperature,
            )

            evaluation = evaluate_response(
                test_case=test_case,
                generated_response=generated,
                judge_provider=judge_provider,
            )

            overall_score = calculate_overall_score(
                evaluation
            )

            save_evaluation(
                run_id=run_id,
                dataset_version=dataset_version,
                prompt_version=prompt_version,
                requested_model=generated.requested_model,
                actual_model=generated.actual_model,
                test_id=generated.test_id,
                response=generated.response,
                factuality=evaluation.factuality,
                relevance=evaluation.relevance,
                format_compliance=evaluation.format_compliance,
                faithfulness=evaluation.faithfulness,
                overall_score=overall_score,
                reason=evaluation.reason,
                judge_model=evaluation.judge_model,
            )

            results.append(evaluation)

        except Exception as error:
            failures.append(
                {
                    "test_id": test_case.id,
                    "error": str(error),
                }
            )

        if progress_callback:
            progress_callback(
                index,
                total,
                test_case.id,
            )

    return {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "prompt_version": prompt_version,
        "total": total,
        "successful": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }