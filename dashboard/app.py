"""Credit Intelligence Dashboard powered by FastAPI v2 and safe aggregates.

Run locally with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.api_client import ApiStatus, CreditApiClient, PredictionResult
from dashboard.components import (
    ANALYTICS_PATH,
    ANALYTICS_SHA256_PATH,
    CLASS_ORDER,
    EDUCATIONAL_DISCLAIMER,
    FINAL_METRICS,
    MODEL_METADATA_PATH,
    NAVIGATION_SECTIONS,
    OPTIONAL_PREDICTION_FIELDS,
    PARTITION_SIZES,
    PER_CLASS_METRICS,
    PREDICTION_FIELDS,
    VALIDATION_MACRO_F1,
    AnalyticsIntegrityError,
    MetadataIntegrityError,
    bivariate_figure,
    build_prediction_payload,
    confusion_matrix_figure,
    correlation_figure,
    extreme_invalid_figure,
    histogram_figure,
    load_model_metadata,
    load_verified_analytics,
    missing_values_figure,
    occupation_figure,
    per_class_metrics_figure,
    prediction_is_available,
    probability_figure,
    target_distribution_figure,
    validation_final_figure,
)
from dashboard.theme import DASHBOARD_CSS


CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


@st.cache_data(show_spinner=False)
def cached_analytics() -> dict[str, Any]:
    return load_verified_analytics(ANALYTICS_PATH, ANALYTICS_SHA256_PATH)


@st.cache_data(show_spinner=False)
def cached_metadata() -> dict[str, Any]:
    return load_model_metadata(MODEL_METADATA_PATH)


def render_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_api_state(status: ApiStatus) -> None:
    if status.ready:
        st.success(
            f"Model API ready · version {status.model_version} · "
            f"artifact {status.artifact_integrity}"
        )
    elif status.available:
        st.warning("The API is reachable, but model v2 is not ready.")
    else:
        st.warning(
            "The API is unavailable. Prediction is paused; analytics remain usable."
        )


def render_executive_overview(
    analytics: dict[str, Any] | None,
    metadata: dict[str, Any],
    readiness: ApiStatus,
) -> None:
    render_page_header(
        "Executive Overview",
        "A concise view of the verified educational credit-classification system.",
    )
    st.info(EDUCATIONAL_DISCLAIMER, icon="ℹ️")

    row_count = f"{analytics['row_count']:,}" if analytics else "Unavailable"
    customer_count = (
        f"{analytics['customer_count']:,}" if analytics else "Unavailable"
    )
    first = st.columns(4)
    first[0].metric("Model version", metadata["model_version"])
    first[1].metric("Selected model", "Decision Tree")
    first[2].metric("Development rows", row_count)
    first[3].metric("Development customers", customer_count)

    second = st.columns(4)
    second[0].metric("Final Macro F1", f"{FINAL_METRICS['macro_f1']:.4f}")
    second[1].metric(
        "Final balanced accuracy",
        f"{FINAL_METRICS['balanced_accuracy']:.4f}",
    )
    second[2].metric(
        "Final Poor recall", f"{FINAL_METRICS['poor_recall']:.4f}"
    )
    second[3].metric(
        "Portfolio criteria",
        "Accepted" if metadata["portfolio_acceptance"] else "Not accepted",
    )

    st.subheader("Live model readiness")
    render_api_state(readiness)

    st.subheader("What this system demonstrates")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Prediction task**

            Classify a monthly financial profile as **Poor**, **Standard**, or
            **Good** using a frozen Decision Tree pipeline.

            **Customer-grouped splitting**

            Every customer’s monthly records stay in one partition, preventing
            the same customer from appearing in both development and evaluation.
            """
        )
    with right:
        st.markdown(
            """
            **Final-test discipline**

            The final test was evaluated exactly once after model selection.
            Its results cannot be used for further tuning.

            **Responsible scope**

            This is an educational architecture and analysis demonstration,
            not a lending decision or regulatory assessment.
            """
        )


def _optional_number(
    label: str,
    key: str,
    *,
    help_text: str,
    integer: bool = False,
) -> float | int | None:
    available = st.checkbox(f"Provide {label.lower()}", value=False, key=f"{key}_available")
    if not available:
        st.caption(f"{label}: unavailable (sent as null)")
        return None
    step = 1 if integer else 0.01
    return st.number_input(
        label,
        value=None,
        step=step,
        help=help_text,
        key=key,
    )


def _required_values_present(values: dict[str, object]) -> bool:
    return all(
        values[field] is not None
        for field in PREDICTION_FIELDS
        if field not in OPTIONAL_PREDICTION_FIELDS
    )


def render_prediction_result(result: PredictionResult) -> None:
    if not result.ok:
        st.error(result.message)
        return

    st.subheader("Prediction result")
    columns = st.columns(4)
    columns[0].metric("Predicted category", result.credit_score)
    columns[1].metric("Risk interpretation", result.risk_level)
    columns[2].metric("Confidence", f"{result.confidence:.1%}")
    columns[3].metric("Model version", result.model_version)
    st.plotly_chart(
        probability_figure(result),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    interpretations = {
        "Poor": (
            "The model assigned the largest probability to the Poor category. "
            "This is a model classification, not a lending recommendation."
        ),
        "Standard": (
            "The model assigned the largest probability to the Standard "
            "category. The result should be interpreted only as a portfolio demo."
        ),
        "Good": (
            "The model assigned the largest probability to the Good category. "
            "It does not establish eligibility or financial suitability."
        ),
    }
    st.info(interpretations[result.credit_score])
    st.warning(EDUCATIONAL_DISCLAIMER)


def render_credit_prediction(
    client: CreditApiClient,
    readiness: ApiStatus,
) -> None:
    render_page_header(
        "Credit Prediction",
        "Submit one synthetic or hypothetical profile to the configured FastAPI v2 endpoint.",
    )
    st.caption(
        f"Inputs are sent only to `{client.base_url}/api/v2/predict`. "
        "They are not added to dashboard history."
    )
    render_api_state(readiness)

    with st.form("credit_prediction_form", clear_on_submit=False):
        st.subheader("Personal context")
        personal = st.columns(2)
        age = personal[0].number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=None,
            step=1,
            help="Whole years; required by the model API.",
        )
        occupation_available = personal[1].checkbox(
            "Provide occupation", value=False
        )
        occupation = (
            personal[1].text_input(
                "Occupation",
                value="",
                max_chars=100,
                help="Optional descriptive category; leave unavailable if unknown.",
            ).strip()
            if occupation_available
            else None
        )
        occupation = occupation or None

        st.subheader("Income and obligations")
        income = st.columns(3)
        annual_income = income[0].number_input(
            "Annual income",
            min_value=0.0,
            value=None,
            step=100.0,
            help="Required. Currency units follow the source dataset and are not documented as USD.",
        )
        with income[1]:
            monthly_salary = _optional_number(
                "Monthly in-hand salary",
                "monthly_inhand_salary",
                help_text="Optional; uses the source dataset’s unspecified currency units.",
            )
        total_emi = income[2].number_input(
            "Total EMI per month",
            min_value=0.0,
            value=None,
            step=10.0,
            help="Required monthly loan-repayment amount in dataset currency units.",
        )

        st.subheader("Credit accounts")
        accounts = st.columns(4)
        bank_accounts = accounts[0].number_input(
            "Number of bank accounts",
            min_value=0,
            value=None,
            step=1,
        )
        credit_cards = accounts[1].number_input(
            "Number of credit cards",
            min_value=0,
            value=None,
            step=1,
        )
        loans = accounts[2].number_input(
            "Number of loans",
            min_value=0,
            value=None,
            step=1,
        )
        with accounts[3]:
            inquiries = _optional_number(
                "Number of credit inquiries",
                "num_credit_inquiries",
                help_text="Optional count of credit inquiries.",
                integer=True,
            )

        st.subheader("Payment history")
        payment = st.columns(3)
        delayed = payment[0].number_input(
            "Number of delayed payments",
            min_value=0,
            value=None,
            step=1,
        )
        with payment[1]:
            changed_limit = _optional_number(
                "Changed credit limit",
                "changed_credit_limit",
                help_text="Optional signed change; dataset units are not specified.",
            )
        history_months = payment[2].number_input(
            "Credit-history age in months",
            min_value=0,
            value=None,
            step=1,
            help="Required duration expressed as whole months.",
        )

        st.subheader("Debt and utilization")
        debt = st.columns(2)
        outstanding_debt = debt[0].number_input(
            "Outstanding debt",
            min_value=0.0,
            value=None,
            step=10.0,
            help="Required; dataset currency units are not documented.",
        )
        utilization = debt[1].number_input(
            "Credit-utilization ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            help="Required percentage from 0 to 100.",
        )

        values = {
            "age": age,
            "annual_income": annual_income,
            "monthly_inhand_salary": monthly_salary,
            "num_bank_accounts": bank_accounts,
            "num_credit_cards": credit_cards,
            "num_loans": loans,
            "num_delayed_payments": delayed,
            "changed_credit_limit": changed_limit,
            "num_credit_inquiries": inquiries,
            "outstanding_debt": outstanding_debt,
            "credit_utilization_ratio": utilization,
            "credit_history_age_months": history_months,
            "total_emi_per_month": total_emi,
            "occupation": occupation,
        }
        submitted = st.form_submit_button(
            "Request prediction",
            type="primary",
            use_container_width=True,
            disabled=not prediction_is_available(readiness),
        )

    if submitted:
        if not _required_values_present(values):
            st.error("Complete every required field before requesting a prediction.")
            return
        payload = build_prediction_payload(values)
        with st.spinner("Requesting one prediction from FastAPI v2…"):
            result = client.predict(payload)
        render_prediction_result(result)


def _largest_bin_takeaway(histogram: dict[str, Any], label: str) -> str:
    index = max(range(len(histogram["percentages"])), key=histogram["percentages"].__getitem__)
    low = histogram["bin_edges"][index]
    high = histogram["bin_edges"][index + 1]
    return (
        f"The largest published {label} bin is {low:,.0f}–{high:,.0f}, "
        f"containing {histogram['percentages'][index]:.1f}% of development rows."
    )


def render_data_insights(analytics: dict[str, Any] | None) -> None:
    render_page_header(
        "Data Insights",
        "Privacy-safe aggregate statistics from development data only.",
    )
    if analytics is None:
        st.error(
            "Analytics are hidden because the artifact could not be verified. "
            "The dashboard will not fall back to raw data."
        )
        return

    threshold = analytics["privacy"]["suppression_threshold"]
    st.info(
        f"All views use cleaned development aggregates only. Bivariate cells "
        f"below k={threshold} are suppressed; no individual records are exposed."
    )

    st.plotly_chart(
        target_distribution_figure(analytics),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    distribution = analytics["target_distribution"]
    largest_class = max(distribution, key=lambda label: distribution[label]["percentage"])
    st.caption(
        f"Takeaway: {largest_class} is the largest development class at "
        f"{distribution[largest_class]['percentage']:.1f}%; percentages avoid "
        "misleading comparisons caused by unequal class sizes."
    )

    histogram_specs = (
        ("Age", "Age distribution", "Age (years)"),
        ("Annual_Income", "Annual-income distribution", "Annual income bin"),
        ("Outstanding_Debt", "Outstanding-debt distribution", "Debt bin"),
        (
            "Credit_Utilization_Ratio",
            "Credit-utilization distribution",
            "Utilization ratio bin (%)",
        ),
    )
    for start in range(0, len(histogram_specs), 2):
        columns = st.columns(2)
        for column, (feature, title, x_title) in zip(
            columns, histogram_specs[start : start + 2], strict=True
        ):
            with column:
                st.plotly_chart(
                    histogram_figure(
                        analytics,
                        feature,
                        title=title,
                        x_title=x_title,
                    ),
                    use_container_width=True,
                    config=CHART_CONFIG,
                )
                st.caption(
                    "Takeaway: "
                    + _largest_bin_takeaway(
                        analytics["histograms"][feature],
                        title.lower(),
                    )
                )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            occupation_figure(analytics),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        st.caption(
            "Takeaway: Unknown represents missing occupation; rare categories "
            "would be grouped as Other rather than exposed separately."
        )
    with right:
        st.plotly_chart(
            missing_values_figure(analytics),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        highest_missing = max(
            analytics["missing_values"],
            key=lambda feature: analytics["missing_values"][feature]["percentage"],
        )
        st.caption(
            f"Takeaway: {highest_missing} has the highest missing share "
            f"({analytics['missing_values'][highest_missing]['percentage']:.1f}%)."
        )

    st.plotly_chart(
        extreme_invalid_figure(analytics),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    st.caption(
        "Takeaway: these are the six fixed audited masking rules; the chart "
        "describes flagged values without asserting why they occurred."
    )

    st.plotly_chart(
        correlation_figure(analytics),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    st.caption(
        "Takeaway: Pearson correlations summarize linear association only; "
        "they do not establish cause and effect."
    )

    st.subheader("Suppressed bivariate aggregates")
    st.caption(
        f"Blank cells contain fewer than {threshold} observations and remain "
        f"suppressed. Total suppressed cells: {analytics['suppressed_bin_count']}."
    )
    for name, title in (
        (
            "Annual_Income_vs_Outstanding_Debt",
            "Annual income versus outstanding debt",
        ),
        (
            "Credit_Utilization_Ratio_vs_Outstanding_Debt",
            "Credit utilization versus outstanding debt",
        ),
    ):
        st.plotly_chart(
            bivariate_figure(analytics["bivariate_bins"][name], title=title),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        st.caption(
            "Takeaway: this view compares aggregate bin shares only; suppressed "
            "cells cannot be used to reconstruct individual profiles."
        )


def render_model_performance() -> None:
    render_page_header(
        "Model Performance",
        "Frozen validation and one-time final-test evidence; no metrics are recomputed here.",
    )
    first_metrics = st.columns(4)
    first_metrics[0].metric(
        "Validation Macro F1", f"{VALIDATION_MACRO_F1:.4f}"
    )
    first_metrics[1].metric(
        "Final Macro F1", f"{FINAL_METRICS['macro_f1']:.4f}"
    )
    first_metrics[2].metric(
        "Final accuracy", f"{FINAL_METRICS['accuracy']:.4f}"
    )
    first_metrics[3].metric(
        "Balanced accuracy", f"{FINAL_METRICS['balanced_accuracy']:.4f}"
    )
    second_metrics = st.columns(4)
    second_metrics[0].metric(
        "Weighted F1", f"{FINAL_METRICS['weighted_f1']:.4f}"
    )
    second_metrics[1].metric(
        "Macro precision", f"{FINAL_METRICS['macro_precision']:.4f}"
    )
    second_metrics[2].metric(
        "Macro recall", f"{FINAL_METRICS['macro_recall']:.4f}"
    )
    second_metrics[3].metric(
        "Poor recall", f"{FINAL_METRICS['poor_recall']:.4f}"
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            per_class_metrics_figure(),
            use_container_width=True,
            config=CHART_CONFIG,
        )
    with right:
        st.plotly_chart(
            confusion_matrix_figure(),
            use_container_width=True,
            config=CHART_CONFIG,
        )
    st.dataframe(
        [
            {
                "Class": label,
                "Precision": PER_CLASS_METRICS[label]["precision"],
                "Recall": PER_CLASS_METRICS[label]["recall"],
                "F1": PER_CLASS_METRICS[label]["f1"],
                "Support": PER_CLASS_METRICS[label]["support"],
            }
            for label in CLASS_ORDER
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.plotly_chart(
        validation_final_figure(),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    st.subheader("How to read these results")
    explanation = st.columns(3)
    explanation[0].markdown(
        "**Macro F1** gives Poor, Standard, and Good equal importance before "
        "averaging class F1 scores."
    )
    explanation[1].markdown(
        "**Poor recall** is the share of actual Poor records correctly identified. "
        "Missed Poor cases remain an important limitation."
    )
    explanation[2].markdown(
        "**Confusion matrix** rows are actual classes and columns are predicted "
        "classes, always ordered Poor, Standard, Good."
    )

    st.subheader("Why the Decision Tree was selected")
    st.markdown(
        """
        - Stable validation performance and a small train–validation gap.
        - Simpler inspection and explanation than the tuned Random Forest.
        - Selection was frozen before the final test was opened.
        """
    )
    st.warning(
        "The final test was evaluated exactly once. These results cannot be "
        "used for further tuning and do not prove fairness, probability "
        "calibration, or readiness for real-world deployment."
    )


def render_model_card(
    analytics: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> None:
    render_page_header(
        "Model Card & Limitations",
        "A transparent summary of intended use, construction, reproducibility, and risk.",
    )
    st.info(EDUCATIONAL_DISCLAIMER)

    overview = st.columns(4)
    overview[0].metric("Model", "Credit Score Decision Tree")
    overview[1].metric("Version", metadata["model_version"])
    overview[2].metric("Raw inputs", len(metadata["raw_model_features"]))
    overview[3].metric(
        "Transformed features", metadata["transformed_feature_count"]
    )

    st.subheader("Frozen technical contract")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            **Algorithm:** DecisionTreeClassifier

            **Hyperparameters:** `max_depth=6`,
            `min_samples_leaf=100`, `class_weight=None`, `random_state=42`

            **Target classes:** {", ".join(CLASS_ORDER)}

            **Split:** customer-grouped, seed 42

            **Labeled dataset:** {PARTITION_SIZES['labeled_dataset']['rows']:,} rows /
            {PARTITION_SIZES['labeled_dataset']['customers']:,} customers

            **Development:** {PARTITION_SIZES['development']['rows']:,} rows /
            {PARTITION_SIZES['development']['customers']:,} customers

            **Validation:** {PARTITION_SIZES['validation']['rows']:,} rows /
            {PARTITION_SIZES['validation']['customers']:,} customers

            **Final test:** {PARTITION_SIZES['final_test']['rows']:,} rows /
            {PARTITION_SIZES['final_test']['customers']:,} customers
            """
        )
    with right:
        st.markdown(
            f"""
            **Deterministic cleaning:** validated numeric parsing and fixed invalidation

            **Extreme handling:** six audited masks plus indicator features

            **Missing values:** development-fitted median imputation

            **Occupation:** explicit Unknown plus one-hot encoding

            **Artifact SHA-256:** `{metadata['model_artifact_sha256']}`
            """
        )

    st.subheader("Fourteen raw model inputs")
    st.write(", ".join(metadata["raw_model_features"]))

    intended, excluded = st.columns(2)
    with intended:
        st.subheader("Intended use")
        st.markdown(
            """
            - Educational demonstration of leakage-safe ML architecture.
            - Synthetic or hypothetical inference through FastAPI v2.
            - Portfolio discussion of grouped evaluation and responsible limits.
            """
        )
    with excluded:
        st.subheader("Out-of-scope uses")
        st.markdown(
            """
            - Real lending approval, pricing, eligibility, or adverse action.
            - Automated decisions about identifiable people.
            - Regulatory, legal, fairness, or credit-bureau compliance claims.
            """
        )

    st.subheader("Known limitations and ethical considerations")
    st.markdown(
        """
        - Educational portfolio project only; not approved for real lending decisions.
        - Dataset provenance, collection context, and currency units are limited.
        - The dataset may be synthetic and may not reflect current populations.
        - Occupation may act as a socioeconomic proxy.
        - Age may require legal, policy, and fairness review before any real use.
        - Probabilities have not been proven calibrated.
        - No external or temporal validation has been completed.
        - No formal bias or fairness audit has been completed.
        - No regulatory validation has been completed.
        - The model does not replace qualified human review.
        """
    )

    st.subheader("Reproducibility summary")
    st.markdown(
        f"""
        The split seed, deterministic cleaner, fixed extreme-value rules, frozen
        preprocessing contract, model metadata, and sanitized development
        analytics are versioned in the repository. The analytics artifact records
        generator commit `{analytics['generator_commit'] if analytics else 'Unavailable'}`
        and is verified against its SHA-256 sidecar before display.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Credit Intelligence Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    client = CreditApiClient()
    readiness = client.readiness()

    try:
        analytics = cached_analytics()
        analytics_error = None
    except AnalyticsIntegrityError as error:
        analytics = None
        analytics_error = str(error)

    try:
        metadata = cached_metadata()
    except MetadataIntegrityError:
        st.error("Verified model metadata are unavailable.")
        st.stop()

    with st.sidebar:
        st.title("Credit Intelligence")
        st.caption("Verified portfolio analytics and API inference")
        section = st.radio(
            "Navigate",
            NAVIGATION_SECTIONS,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Configured FastAPI")
        st.code(client.base_url, language=None)
        render_api_state(readiness)
        st.caption("No applicant history is retained by this dashboard.")

    if analytics_error and section in {"Executive Overview", "Data Insights"}:
        st.warning(analytics_error)

    if section == "Executive Overview":
        render_executive_overview(analytics, metadata, readiness)
    elif section == "Credit Prediction":
        render_credit_prediction(client, readiness)
    elif section == "Data Insights":
        render_data_insights(analytics)
    elif section == "Model Performance":
        render_model_performance()
    else:
        render_model_card(analytics, metadata)


if __name__ == "__main__":
    main()
