import os

from dotenv import load_dotenv


load_dotenv()


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    return api_key


def get_model() -> str:
    model = os.getenv("LLM_MODEL")

    if not model:
        raise ValueError(
            "LLM_MODEL is not configured."
        )

    return model