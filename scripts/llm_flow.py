from prompts.loader import load_prompt
from llm.provider import OpenRouterProvider


user_input = "Explain what an API is."

provider = OpenRouterProvider()

for version in ["v1", "v2"]:

    prompt = load_prompt(
        version,
        user_input,
    )

    response = provider.generate(
        prompt,
        temperature=0.0,
    )

    print(f"\n--- Prompt {version} ---")
    print(response.text)