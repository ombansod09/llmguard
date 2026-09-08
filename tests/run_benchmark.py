from datasets.loader import load_dataset
from llm.provider import OpenRouterProvider
from llm.runner import generate_response


DATASET_VERSION = "benchmark_v1"
PROMPT_VERSION = "v1"


def main():

    dataset = load_dataset(DATASET_VERSION)

    provider = OpenRouterProvider()

    print(f"Dataset: {dataset.version}")
    print(f"Tests: {len(dataset.tests)}")
    print(f"Prompt: {PROMPT_VERSION}")
    print(f"Model: {provider.get_model_name()}")

    print("\nGenerating responses...\n")

    responses = []

    for index, test_case in enumerate(
        dataset.tests,
        start=1,
    ):
        print(
            f"[{index}/{len(dataset.tests)}] "
            f"{test_case.id}"
        )

        result = generate_response(
            test_case=test_case,
            prompt_version=PROMPT_VERSION,
            provider=provider,
            temperature=0.0,
        )

        responses.append(result)

        print(f"Response: {result.response[:150]}")
        print()

    print(
        f"Generated {len(responses)} responses successfully."
    )


if __name__ == "__main__":
    main()