from llm.provider import OpenRouterProvider


provider = OpenRouterProvider()

response = provider.generate(
    "What is the capital of France?"
)

print("Model:", provider.get_model_name())
print("Response:", response)