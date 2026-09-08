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

    for index, test_case in enumerate(dataset.tests, start=1):

        print(
            f"[{index}/{len(dataset.tests)}] "
            f"{test_case.id}"
        )

        try:
            result = generate_response(
                test_case=test_case,
                prompt_version=PROMPT_VERSION,
                provider=provider,
                temperature=0.0,
            )

            responses.append(result)

            print(
                f"Actual model: {result.actual_model}"
            )

            print(
                f"Response: "
                f"{result.response[:150]}"
            )

            print()

        except Exception as error:
            print(
                f"ERROR generating "
                f"{test_case.id}: {error}"
            )

    successful = len(responses)
    failed = len(dataset.tests) - successful

    print("\nGeneration Summary")
    print("------------------")
    print(f"Total:      {len(dataset.tests)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")

    if failed > 0:
        print(
            "\nWARNING: Some responses "
            "were not generated."
        )
    else:
        print(
            "\nAll responses generated successfully."
        )


if __name__ == "__main__":
    main()