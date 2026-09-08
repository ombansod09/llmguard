from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    test_id: str
    factuality: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    format_compliance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)

    reason: str = Field(..., description="Reason for the evaluation result")
    judge_model: str = Field(..., description="Model used for evaluation")