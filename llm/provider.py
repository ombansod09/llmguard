from openai import OpenAI

from config.settings import get_api_key, get_model
from llm.base import LLMProvider
from llm.models import LLMResponse


class OpenRouterProvider(LLMProvider):

    def __init__(self, model: str | None = None):
        self.model = model or get_model()

        self.client = OpenAI(
            api_key=get_api_key(),
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> LLMResponse:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
        )

        text = response.choices[0].message.content

        if not text:
            raise ValueError("LLM returned an empty response.")

        actual_model = response.model

        if not actual_model:
            raise ValueError(
                "OpenRouter did not return the actual model."
            )

        return LLMResponse(
            text=text,
            requested_model=self.model,
            actual_model=actual_model,
        )

    def get_model_name(self) -> str:
        return self.model