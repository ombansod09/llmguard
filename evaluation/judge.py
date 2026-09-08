import json
import re

from llm.base import LLMProvider


JUDGE_PROMPT = """
You are an expert evaluator of LLM responses.

Evaluate the response against the user's question, expected answer,
and provided context.

Score the response on four dimensions:

1. factuality
   - Is the information factually correct?
   - Does it agree with the expected answer where applicable?

2. relevance
   - Does the response directly answer the user's question?
   - Is unnecessary or unrelated information avoided?

3. format_compliance
   - Did the response follow the requested format and constraints?
   - Examples include exact number of items, JSON, XML, tables,
     numbered lists, and sentence limits.

4. faithfulness
   - Is the response supported by the provided context?
   - If context is provided, penalize claims that contradict or
     go beyond the context.
   - If no context is provided, evaluate whether the response
     introduces unsupported claims relative to the task.

Scores must be numbers between 0 and 1.

Return ONLY valid JSON.
Do not use markdown.
Do not include any text before or after the JSON.

Required JSON format:

{{
  "factuality": 0.0,
  "relevance": 0.0,
  "format_compliance": 0.0,
  "faithfulness": 0.0,
  "reason": "Brief explanation of the scores."
}}

USER QUESTION:
{input}

EXPECTED ANSWER:
{expected}

CONTEXT:
{context}

LLM RESPONSE:
{response}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Judge did not return valid JSON."
        )

    return json.loads(match.group(0))

def judge_response(
    test_case,
    response_text: str,
    provider: LLMProvider,
):
    prompt = JUDGE_PROMPT.format(
        input=test_case.input,
        expected=test_case.expected,
        context=test_case.context or "None provided",
        response=response_text,
    )

    result = provider.generate(
        prompt,
        temperature=0.0,
    )

    try:
        scores = _extract_json(result.text)

    except (json.JSONDecodeError, ValueError):
        retry_prompt = f"""
Your previous response was not valid JSON.

Return ONLY valid JSON using exactly this structure:

{{
  "factuality": 0.0,
  "relevance": 0.0,
  "format_compliance": 0.0,
  "faithfulness": 0.0,
  "reason": "Brief explanation of the scores."
}}

Do not use markdown.
Do not include any text before or after the JSON.

Original evaluation request:

{prompt}
"""

        retry_result = provider.generate(
            retry_prompt,
            temperature=0.0,
        )

        scores = _extract_json(
            retry_result.text
        )

        result = retry_result

    values = {
        "factuality": float(
            scores["factuality"]
        ),
        "relevance": float(
            scores["relevance"]
        ),
        "format_compliance": float(
            scores["format_compliance"]
        ),
        "faithfulness": float(
            scores["faithfulness"]
        ),
    }

    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Judge score for {name} "
                f"must be between 0 and 1. "
                f"Received: {value}"
            )

    return {
        **values,
        "reason": str(scores["reason"]),
        "judge_model": result.actual_model,
    }