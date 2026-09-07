from pathlib import Path


PROMPTS_DIR = Path(__file__).parent


def load_prompt(version: str, user_input: str) -> str:
    prompt_file = PROMPTS_DIR / f"prompt_{version}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt version not found: {version}"
        )

    template = prompt_file.read_text(encoding="utf-8")

    return template.format(input=user_input)