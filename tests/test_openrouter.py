from openai import OpenAI

from config.settings import get_api_key, get_model


client = OpenAI(
    api_key=get_api_key(),
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model=get_model(),
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    temperature=0,
)

print(response.choices[0].message.content)