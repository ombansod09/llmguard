import os

from dotenv import load_dotenv


load_dotenv()


def _get_config_value(name: str) -> str | None:
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        value = None

    return value


def get_api_key() -> str:
    api_key = _get_config_value("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    return api_key


def get_model() -> str:
    model = _get_config_value("LLM_MODEL")

    if not model:
        raise ValueError(
            "LLM_MODEL is not configured."
        )

    return model
