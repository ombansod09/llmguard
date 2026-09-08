from datasets.loader import load_dataset
from evaluation.evaluator import evaluate_response
from llm.provider import OpenRouterProvider
from llm.runner import generate_response


def main():
    dataset = load_dataset("benchmark_v1")

    generation_provider = OpenRouterProvider()
    judge_provider = OpenRouterProvider()

    test_case = dataset.tests[0]

    generated = generate_response(
        test_case=test_case,
        prompt_version="v1",
        provider=generation_provider,
        temperature=0.0,
    )

    evaluation = evaluate_response(
        test_case=test_case,
        generated_response=generated,
        judge_provider=judge_provider,
    )

    print("Test ID:", evaluation.test_id)
    print()
    print("Factuality:", evaluation.factuality)
    print("Relevance:", evaluation.relevance)
    print(
        "Format:",
        evaluation.format_compliance,
    )
    print(
        "Faithfulness:",
        evaluation.faithfulness,
    )
    print("Judge model:", evaluation.judge_model)
    print("Reason:", evaluation.reason)


if __name__ == "__main__":
    main()