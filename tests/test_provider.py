from llm.provider import OpenRouterProvider


def main():
    provider = OpenRouterProvider()

    response = provider.generate(
        "Say hello in one sentence.",
        temperature=0.0,
    )

    print("Requested model:", response.requested_model)
    print("Actual model:", response.actual_model)
    print("Response:", response.text)


if __name__ == "__main__":
    main()