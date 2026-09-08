from llm.base import LLMProvider
from llm.models import GeneratedResponse
from prompts.loader import load_prompt


def generate_response(
    test_case,
    prompt_version: str,
    provider: LLMProvider,
    temperature: float = 0.0,
) -> GeneratedResponse:

    prompt = load_prompt(
        prompt_version,
        test_case.input,
    )

    response = provider.generate(
        prompt,
        temperature=temperature,
    )

    return GeneratedResponse(
        test_id=test_case.id,
        prompt_version=prompt_version,
        requested_model=response.requested_model,
        actual_model=response.actual_model,
        input=test_case.input,
        expected=test_case.expected,
        context=test_case.context,
        response=response.text,
    )