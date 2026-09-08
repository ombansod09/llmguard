from datasets.loader import load_dataset
from llm.provider import OpenRouterProvider
from llm.runner import generate_response


dataset = load_dataset("benchmark_v1")
provider = OpenRouterProvider()

test_case = dataset.tests[0]

result = generate_response(
    test_case=test_case,
    prompt_version="v1",
    provider=provider,
    temperature=0.0,
)

print("Test ID:", result.test_id)
print("Model:", result.model)
print("Input:", result.input)
print("Expected:", result.expected)
print("Response:", result.response)