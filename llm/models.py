from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    requested_model: str
    actual_model: str


class GeneratedResponse(BaseModel):
    test_id: str
    prompt_version: str
    requested_model: str
    actual_model: str
    input: str
    expected: str
    context: str | None
    response: str