from datasets.loader import load_dataset
from evaluation.evaluator import evaluate_response
from llm.provider import OpenRouterProvider
from llm.runner import generate_response


DATASET_VERSION = "benchmark_v1"
PROMPT_VERSION = "v1"


def main():
    dataset = load_dataset(DATASET_VERSION)

    generation_provider = OpenRouterProvider()
    judge_provider = OpenRouterProvider()

    print(f"Dataset: {dataset.version}")
    print(f"Tests: {len(dataset.tests)}")
    print(f"Prompt: {PROMPT_VERSION}")
    print(
        f"Generation model: "
        f"{generation_provider.get_model_name()}"
    )
    print(
        f"Judge model: "
        f"{judge_provider.get_model_name()}"
    )

    print("\nRunning evaluation...\n")

    results = []

    for index, test_case in enumerate(
        dataset.tests,
        start=1,
    ):
        print(
            f"[{index}/{len(dataset.tests)}] "
            f"{test_case.id}"
        )

        try:
            generated = generate_response(
                test_case=test_case,
                prompt_version=PROMPT_VERSION,
                provider=generation_provider,
                temperature=0.0,
            )

            evaluation = evaluate_response(
                test_case=test_case,
                generated_response=generated,
                judge_provider=judge_provider,
            )

            results.append(evaluation)

            print(
                f"Generation model: "
                f"{generated.actual_model}"
            )

            print(
                f"Judge model: "
                f"{evaluation.judge_model}"
            )

            print(
                f"Scores: "
                f"F={evaluation.factuality:.2f}, "
                f"R={evaluation.relevance:.2f}, "
                f"Fmt={evaluation.format_compliance:.2f}, "
                f"Faith={evaluation.faithfulness:.2f}"
            )

            print()

        except Exception as error:
            print(
                f"ERROR evaluating "
                f"{test_case.id}: {error}"
            )

    successful = len(results)
    failed = len(dataset.tests) - successful

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total:      {len(dataset.tests)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")

    if failed == 0:
        print("\nAll evaluations completed successfully.")
    else:
        print("\nWARNING: Some evaluations failed.")


if __name__ == "__main__":
    main()