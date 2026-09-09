import streamlit as st
from types import SimpleNamespace

from datasets.loader import load_dataset
from evaluation.experiment import run_experiment
from regression.detector import detect_regression
from regression.analyzer import find_regressions
from storage.database import (
    initialize_database,
    get_run_ids,
    get_experiment_results,
)


st.set_page_config(
    page_title="LLMGuard",
    page_icon="🛡️",
    layout="wide",
)


initialize_database()


st.title("🛡️ LLMGuard")
st.caption(
    "LLM Response Evaluation & Prompt Regression Framework"
)


run_tab, results_tab, compare_tab = st.tabs(
    [
        "Run Evaluation",
        "Results",
        "Compare",
    ]
)


# ---------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------

with run_tab:

    st.header("Run Evaluation")

    dataset_version = st.selectbox(
        "Benchmark Dataset",
        ["benchmark_v1"],
    )

    prompt_version = st.selectbox(
        "Prompt Version",
        ["v1", "v2"],
    )

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

    st.divider()

    dataset = load_dataset(dataset_version)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Test Cases",
            len(dataset.tests),
        )

    with col2:
        st.metric(
            "Prompt",
            prompt_version,
        )

    if st.button(
        "Run Benchmark",
        type="primary",
        use_container_width=True,
    ):

        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(
            current,
            total,
            test_id,
        ):
            progress_bar.progress(
                current / total
            )

            status.write(
                f"Evaluating {test_id} "
                f"({current}/{total})"
            )

        with st.spinner(
            "Running benchmark evaluation..."
        ):
            experiment = run_experiment(
                dataset_version=dataset_version,
                prompt_version=prompt_version,
                temperature=temperature,
                progress_callback=update_progress,
            )

        status.empty()

        if experiment["failed"] == 0:

            st.success(
                f"Evaluation completed successfully. "
                f"{experiment['successful']}/"
                f"{experiment['total']} tests passed."
            )

        else:

            st.warning(
            f"Evaluation completed with "
                f"{experiment['successful']} successful "
                f"and {experiment['failed']} failed tests."
            )

            if experiment["failures"]:

                with st.expander("View failures"):

                    for failure in experiment["failures"]:

                        st.write(
                            f"**{failure['test_id']}**: "
                            f"{failure['error']}"
                        )

        st.write(
            f"Run ID: `{experiment['run_id']}`"
        )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

    with results_tab:

        st.header("Results")

        runs = get_run_ids()

        if not runs:

            st.info(
                "No evaluation runs have been completed yet."
            )

        else:

            run_options = {
                (
                    f"{run['prompt_version']} | "
                    f"{run['created_at']} | "
                    f"{run['run_id'][:8]}"
                ): run["run_id"]
                for run in runs
            }

            selected_label = st.selectbox(
                "Select Evaluation Run",
                list(run_options.keys()),
            )

            selected_run_id = run_options[selected_label]

            rows = get_experiment_results(
                run_id=selected_run_id
            )

            if rows:

                first_row = rows[0]

                st.subheader("Run Summary")

                col1, col2, col3, col4, col5 = st.columns(5)

                overall_score = sum(
                    row["overall_score"]
                    for row in rows
                ) / len(rows)

                factuality = sum(
                    row["factuality"]
                    for row in rows
                ) / len(rows)

                relevance = sum(
                    row["relevance"]
                    for row in rows
                ) / len(rows)

                format_compliance = sum(
                    row["format_compliance"]
                    for row in rows
                ) / len(rows)

                faithfulness = sum(
                    row["faithfulness"]
                    for row in rows
                ) / len(rows)

                with col1:
                    st.metric(
                        "Overall",
                        f"{overall_score * 100:.2f}%",
                    )

                with col2:
                    st.metric(
                        "Factuality",
                        f"{factuality * 100:.2f}%",
                    )

                with col3:
                    st.metric(
                        "Relevance",
                        f"{relevance * 100:.2f}%",
                    )

                with col4:
                    st.metric(
                        "Format",
                        f"{format_compliance * 100:.2f}%",
                    )

                with col5:
                    st.metric(
                        "Faithfulness",
                        f"{faithfulness * 100:.2f}%",
                    )

                st.divider()

                st.subheader("Run Details")

                detail_col1, detail_col2, detail_col3 = (
                    st.columns(3)
                )

                with detail_col1:
                    st.write(
                        f"**Dataset:** "
                        f"{first_row['dataset_version']}"
                    )

                    st.write(
                        f"**Prompt:** "
                        f"{first_row['prompt_version']}"
                    )

                with detail_col2:
                    st.write(
                        f"**Requested Model:** "
                        f"{first_row['requested_model']}"
                    )

                    st.write(
                        f"**Tests:** "
                        f"{len(rows)}"
                    )

                with detail_col3:
                    st.write(
                        f"**Actual Model:** "
                        f"{first_row['actual_model']}"
                    )

                    st.write(
                        f"**Judge Model:** "
                        f"{first_row['judge_model']}"
                    )

                st.divider()

                st.subheader("Per-Test Results")

                table_data = []

                for row in rows:

                    table_data.append(
                        {
                            "Test ID": row["test_id"],
                            "Factuality": (
                                f"{row['factuality'] * 100:.1f}%"
                            ),
                            "Relevance": (
                                f"{row['relevance'] * 100:.1f}%"
                            ),
                            "Format": (
                                f"{row['format_compliance'] * 100:.1f}%"
                            ),
                            "Faithfulness": (
                                f"{row['faithfulness'] * 100:.1f}%"
                            ),
                            "Overall": (
                                f"{row['overall_score'] * 100:.1f}%"
                            ),
                        }
                    )

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                st.subheader("Evaluation Reasons")

                for row in rows:

                    with st.expander(
                        f"{row['test_id']} "
                        f"• {row['overall_score'] * 100:.1f}%"
                    ):

                        st.write(
                            f"**Response:** "
                            f"{row['response']}"
                        )

                        st.write(
                            f"**Reason:** "
                            f"{row['reason']}"
                        )


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

    with compare_tab:

        st.header("Compare")

        runs = get_run_ids()

        if len(runs) < 2:

            st.info(
                "At least two evaluation runs are required "
                "to compare prompt versions."
            )

        else:

            run_labels = {}

            for run in runs:

                label = (
                    f"{run['prompt_version']} | "
                    f"{run['created_at']} | "
                    f"{run['run_id'][:8]}"
                )

                run_labels[label] = run["run_id"]

            labels = list(run_labels.keys())

            st.subheader("Select Runs")

            col1, col2 = st.columns(2)

            with col1:

                before_label = st.selectbox(
                    "Baseline Run",
                    labels,
                    key="compare_before",
                )

            with col2:

                after_label = st.selectbox(
                    "New Run",
                    labels,
                    index=min(1, len(labels) - 1),
                    key="compare_after",
                )

            before_run_id = run_labels[before_label]
            after_run_id = run_labels[after_label]

            if before_run_id == after_run_id:

                st.warning(
                    "Select two different evaluation runs."
                )

            else:

                before_rows = get_experiment_results(
                    run_id=before_run_id
                )

                after_rows = get_experiment_results(
                    run_id=after_run_id
                )

                before_models = sorted(
                    {
                        row["actual_model"]
                        for row in before_rows
                    }
                )

                after_models = sorted(
                    {
                        row["actual_model"]
                        for row in after_rows
                    }
                )

                if before_models != after_models:

                    st.warning(
                        "⚠️ The underlying model selection differs "
                        "between these runs. Regression results should "
                        "not be attributed to the prompt alone."
                    )

                else:

                    st.success(
                        "Underlying generation model selection "
                        "is consistent between these runs."
                    )

                    if not before_rows or not after_rows:

                        st.warning(
                            "One or both selected runs contain no results."
                        )

                    else:

                        before = SimpleNamespace(
                            factuality=sum(
                                row["factuality"]
                                for row in before_rows
                            ) / len(before_rows),

                            relevance=sum(
                                row["relevance"]
                                for row in before_rows
                            ) / len(before_rows),

                            format_compliance=sum(
                                row["format_compliance"]
                                for row in before_rows
                            ) / len(before_rows),

                            faithfulness=sum(
                                row["faithfulness"]
                                for row in before_rows
                            ) / len(before_rows),

                            overall_score=sum(
                                row["overall_score"]
                                for row in before_rows
                            ) / len(before_rows),
                        )

                        after = SimpleNamespace(
                            factuality=sum(
                                row["factuality"]
                                for row in after_rows
                            ) / len(after_rows),

                            relevance=sum(
                                row["relevance"]
                                for row in after_rows
                            ) / len(after_rows),

                            format_compliance=sum(
                                row["format_compliance"]
                                for row in after_rows
                        ) / len(after_rows),

                            faithfulness=sum(
                                row["faithfulness"]
                                for row in after_rows
                            ) / len(after_rows),

                            overall_score=sum(
                                row["overall_score"]
                                for row in after_rows
                            ) / len(after_rows),
                        )

                        regression = detect_regression(
                            before,
                            after,
                        )

                        st.divider()

                        if regression.regression_detected:

                            st.error(
                                "⚠️ Regression detected"
                            )

                        else:

                            st.success(
                                "✅ No regression detected"
                            )

                        st.subheader("Score Comparison")

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Baseline Overall",
                                f"{regression.overall_before * 100:.2f}%",
                            )

                        with col2:

                            st.metric(
                                "New Overall",
                                f"{regression.overall_after * 100:.2f}%",
                            )

                        with col3:

                            st.metric(
                                "Change",
                                f"{regression.overall_change * 100:+.2f} pp",
                            )

                        st.divider()

                        st.subheader("Metric Comparison")

                        metric_table = []

                        for change in regression.metric_changes:

                            metric_table.append(
                                {
                                    "Metric": change.metric,
                                    "Baseline": (
                                        f"{change.before * 100:.1f}%"
                                    ),
                                    "New": (
                                        f"{change.after * 100:.1f}%"
                                    ),
                                    "Change": (
                                        f"{change.change * 100:+.1f} pp"
                                    ),
                                    "Regression": (
                                        "Yes"
                                        if change.change <= -0.10
                                        else "No"
                                    ),
                                }
                            )

                        st.dataframe(
                            metric_table,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.divider()

                        st.subheader(
                            "Per-Test Regression Analysis"
                        )

                        before_analysis = [
                            SimpleNamespace(
                                test_id=row["test_id"],
                                overall_score=row["overall_score"],
                                reason=row["reason"],
                            )
                            for row in before_rows
                        ]

                        after_analysis = [
                            SimpleNamespace(
                                test_id=row["test_id"],
                                overall_score=row["overall_score"],
                                reason=row["reason"],
                            )
                            for row in after_rows
                        ]

                        test_regressions = find_regressions(
                            before_analysis,
                            after_analysis,
                        )

                        if not test_regressions:

                            st.success(
                                "No individual test regressed "
                                "by 5 percentage points or more."
                            )

                        else:

                            st.warning(
                                f"{len(test_regressions)} test(s) "
                                "showed significant regression."
                            )

                            for item in test_regressions:
                            
                                change_pp = (
                                    item.overall_change * 100
                                )

                                with st.expander(
                                    f"{item.test_id} "
                                    f"• {change_pp:+.1f} pp"
                                ):

                                    st.write(
                                        f"**Baseline:** "
                                        f"{item.overall_before * 100:.1f}%"
                                    )

                                    st.write(
                                        f"**New:** "
                                        f"{item.overall_after * 100:.1f}%"
                                    )

                                    st.write(
                                        "**Baseline reason:**"
                                    )
    
                                    st.write(
                                        item.reason_before
                                    )

                                    st.write(
                                        "**New reason:**"
                                    )

                                    st.write(
                                        item.reason_after
                                    )