from pydantic import BaseModel


class MetricChange(BaseModel):
    metric: str
    before: float
    after: float
    change: float


class RegressionResult(BaseModel):
    overall_before: float
    overall_after: float
    overall_change: float

    metric_changes: list[MetricChange]

    overall_regression: bool
    metric_regression: bool
    regression_detected: bool