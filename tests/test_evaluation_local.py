from datasets.loader import load_dataset
from evaluation.evaluator import evaluate_response
from llm.models import GeneratedResponse


class FakeJudgeProvider:

    def generate(self, prompt: str, temperature: float = 0.0):
        class FakeResponse:
            text = """
            {
                "factuality": 0.9,
                "relevance": 0.8,
                "format_compliance": 1.0,
                "faithfulness": 0.7,
                "reason": "Local fake judge response."
            }
            """
            actual_model = "fake-judge-model"

        return FakeResponse()


def main():
    dataset = load_dataset("benchmark_v1")
    test_case = dataset.tests[0]

    generated_response = GeneratedResponse(
        test_id=test_case.id,
        prompt_version="v1",
        requested_model="openrouter/free",
        actual_model="fake-generation-model",
        input=test_case.input,
        expected=test_case.expected,
        context=test_case.context,
        response="Paris.",
    )

    fake_judge = FakeJudgeProvider()

    result = evaluate_response(
        test_case=test_case,
        generated_response=generated_response,
        judge_provider=fake_judge,
    )

    print("Local evaluation test passed.")
    print()
    print("Test ID:", result.test_id)
    print("Factuality:", result.factuality)
    print("Relevance:", result.relevance)
    print("Format:", result.format_compliance)
    print("Faithfulness:", result.faithfulness)
    print("Judge model:", result.judge_model)
    print("Reason:", result.reason)

    assert result.test_id == "fact_001"

    assert result.factuality == 0.9
    assert result.relevance == 0.8
    assert result.format_compliance == 1.0
    assert result.faithfulness == 0.7

    assert result.judge_model == "fake-judge-model"

    assert 0.0 <= result.factuality <= 1.0
    assert 0.0 <= result.relevance <= 1.0
    assert 0.0 <= result.format_compliance <= 1.0
    assert 0.0 <= result.faithfulness <= 1.0

    print()
    print("All assertions passed.")


if __name__ == "__main__":
    main()