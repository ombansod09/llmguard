from regression.models import (
    MetricChange,
    RegressionResult,
)


OVERALL_THRESHOLD = 0.05
METRIC_THRESHOLD = 0.10


def detect_regression(
    before,
    after,
) -> RegressionResult:

    overall_change = round(
        after.overall_score - before.overall_score,
        4,
    )

    metrics = [
        ("factuality", before.factuality, after.factuality),
        ("relevance", before.relevance, after.relevance),
        (
            "format_compliance",
            before.format_compliance,
            after.format_compliance,
        ),
        (
            "faithfulness",
            before.faithfulness,
            after.faithfulness,
        ),
    ]

    metric_changes = []

    for name, before_value, after_value in metrics:

        change = round(
            after_value - before_value,
            4,
        )

        metric_changes.append(
            MetricChange(
                metric=name,
                before=before_value,
                after=after_value,
                change=change,
            )
        )

    overall_regression = (
        overall_change <= -OVERALL_THRESHOLD
    )

    metric_regression = any(
        change.change <= -METRIC_THRESHOLD
        for change in metric_changes
    )

    return RegressionResult(
        overall_before=before.overall_score,
        overall_after=after.overall_score,
        overall_change=overall_change,
        metric_changes=metric_changes,
        overall_regression=overall_regression,
        metric_regression=metric_regression,
        regression_detected=(
            overall_regression
            or metric_regression
        ),
    )