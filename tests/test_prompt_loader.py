from prompts.loader import load_prompt


prompt = load_prompt(
    "v1",
    "What is the capital of France?"
)

assert "What is the capital of France?" in prompt

print("Prompt loading successful.")
print(prompt)