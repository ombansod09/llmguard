def evaluate_deterministic(
    test_case,
    response_text: str,
) -> dict:
    """
    Evaluate benchmark cases using deterministic checks.

    This function is available for benchmark cases explicitly
    marked as deterministic.
    """

    if not response_text.strip():
        return {
            "factuality": 0.0,
            "relevance": 0.0,
            "format_compliance": 0.0,
            "faithfulness": 0.0,
            "reason": "Response is empty.",
        }

    raise NotImplementedError(
        "No deterministic rule is defined for "
        f"test case: {test_case.id}"
    )