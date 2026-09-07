from typing import List, Literal, Optional

from pydantic import BaseModel


class TestCase(BaseModel):
    id: str
    category: Literal[
        "factuality",
        "relevance",
        "format",
        "faithfulness",
    ]
    input: str
    expected: str
    context: Optional[str] = None
    evaluation_type: Literal[
        "llm_judge",
        "deterministic",
    ]


class BenchmarkDataset(BaseModel):
    version: str
    description: str
    tests: List[TestCase]

class ExperimentConfig(BaseModel):
    prompt_version: str
    model: str
    dataset_version: str
    temperature: float = 0.0