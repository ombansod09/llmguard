from abc import ABC, abstractmethod

from llm.models import LLMResponse

class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        raise NotImplementedError