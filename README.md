# 🛡️ LLMGuard

**LLM Response Evaluation & Prompt Regression Framework**

LLMGuard is a lightweight evaluation framework for testing LLM responses against a fixed benchmark, scoring response quality across multiple dimensions, storing experiments, and detecting prompt regressions.

## Why this project?

Prompt changes can improve one type of response while quietly damaging another. LLMGuard turns prompt evaluation into a repeatable experiment instead of relying on a few manual examples.

## Core Pipeline

```text
Benchmark Dataset
      ↓
Prompt Version + Model Configuration
      ↓
LLM Response Generation
      ↓
Response Evaluation
      ↓
Score Aggregation
      ↓
Regression Detection
      ↓
Evaluation Report / UI
```

## Features

- Fixed 30-case benchmark dataset (`benchmark_v1`)
- Prompt versioning with `v1` and `v2`
- OpenRouter provider abstraction
- Standardized response generation and model tracking
- Four evaluation dimensions:
  - Factuality
  - Relevance
  - Format / instruction compliance
  - Faithfulness / hallucination control
- LLM-as-a-judge evaluation with controlled JSON parsing and one retry
- Deterministic evaluation extension point
- Weighted overall score:
  - Factuality: 30%
  - Relevance: 25%
  - Format: 20%
  - Faithfulness: 25%
- Regression thresholds:
  - Overall score decrease ≥ 5 percentage points
  - Any individual metric decrease ≥ 10 percentage points
- Per-test regression analysis with before/after reasons
- SQLite experiment persistence
- Streamlit UI with:
  - Run Evaluation
  - Results
  - Compare
- Model-consistency validation before attributing differences to prompts

## Current Evaluation Status

The application and evaluation pipeline are implemented through the regression-analysis stage. During the final free-tier run, OpenRouter request limits prevented completing a full 30/30 benchmark comparison, so the current demo state should be described as a **working evaluation framework with quota-limited benchmark results**, not as a completed statistical comparison of V1 vs V2.

The observed V2 run completed **26/30 tests**, with an average overall score of **91.91%** across successful evaluations. The four failed cases were caused by OpenRouter's free-model request limit rather than benchmark logic.

Because the `openrouter/free` router can select different underlying models, LLMGuard deliberately warns when two runs use different actual generation models instead of incorrectly attributing the difference to the prompt.

## Project Structure

```text
llmguard/
├── config/          # Configuration and benchmark schemas
├── datasets/        # Versioned benchmark datasets
├── evaluation/      # Judge, evaluation and score aggregation
├── llm/             # Provider abstraction and response generation
├── prompts/         # Versioned prompts
├── regression/      # Regression detection and per-test analysis
├── storage/         # SQLite experiment persistence
├── tests/            # Unit and local evaluation tests
├── app.py            # Streamlit application
├── requirements.txt
└── README.md
```

## Evaluation Model

The framework separates **generation** from **evaluation**. A generated response is evaluated against the benchmark test case, producing normalized scores from 0 to 1. The weighted score is then persisted with the requested model, actual model, judge model, response, and evaluation reason.

Regression analysis compares two stored runs. The application checks the actual generation-model selections first. If they differ, it does not present the result as a prompt-only regression.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ombansod09/llmguard.git
cd llmguard
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
LLM_MODEL=google/gemma-4-26b-a4b-it:free
```

Never commit `.env` or an API key.

### 5. Start the application

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

The app can be deployed from the GitHub repository using Streamlit Community Cloud.

Add these secrets in the deployment settings:

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
LLM_MODEL = "google/gemma-4-26b-a4b-it:free"
```

The configuration layer reads normal environment variables locally and Streamlit secrets when deployed.

## Testing

The project includes local tests for benchmark validation, prompt loading, scoring, storage, regression detection, and per-test analysis.

Some legacy integration scripts intentionally make real OpenRouter calls. They can fail when the provider's free request quota is exhausted. This is an external provider limitation, not evidence that the local evaluation logic is broken.

## Scope

LLMGuard intentionally does **not** include RAG, vector databases, fine-tuning, multiple providers, agents, multimodal features, authentication, cloud infrastructure, Kubernetes, CI/CD, real-time monitoring, notifications, or model training.

The goal is a focused engineering system for **LLM response evaluation and prompt regression testing**.

## Tech Stack

- Python
- Pydantic
- OpenAI-compatible OpenRouter API
- Streamlit
- SQLite
- pytest
- python-dotenv

## Author

Om Bansod  
GitHub: https://github.com/ombansod09
