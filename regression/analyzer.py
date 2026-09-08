from pydantic import BaseModel


class TestRegression(BaseModel):
    test_id: str
    overall_before: float
    overall_after: float
    overall_change: float
    reason_before: str
    reason_after: str


def analyze_test_regression(
    before,
    after,
) -> TestRegression:

    return TestRegression(
        test_id=after.test_id,
        overall_before=before.overall_score,
        overall_after=after.overall_score,
        overall_change=round(
            after.overall_score - before.overall_score,
            4,
        ),
        reason_before=before.reason,
        reason_after=after.reason,
    )


def find_regressions(
    before_results,
    after_results,
) -> list[TestRegression]:

    before_by_id = {
        result.test_id: result
        for result in before_results
    }

    regressions = []

    for after in after_results:

        before = before_by_id.get(after.test_id)

        if before is None:
            continue

        change = after.overall_score - before.overall_score

        if change <= -0.05:
            regressions.append(
                analyze_test_regression(
                    before,
                    after,
                )
            )

    regressions.sort(
        key=lambda result: result.overall_change
    )

    return regressions