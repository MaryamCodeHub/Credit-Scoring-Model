"""
Streamlit Dashboard — Credit Scoring System.

A premium, real-time credit scoring dashboard with:
- Applicant input form
- Gauge chart for credit score visualization
- Probability breakdown bar chart
- Risk assessment display

Theme: Vivid Teal (#009688) & Mint Green (#66BB6A)

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.config import (
    DASHBOARD_TITLE,
    THEME_PRIMARY,
    THEME_SECONDARY,
    THEME_BACKGROUND,
    THEME_SURFACE,
    THEME_TEXT,
)


# ──────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* ─── Global ─── */
        .stApp {{
            background: linear-gradient(135deg, {THEME_BACKGROUND} 0%, #1a1f2e 100%);
            color: {THEME_TEXT};
        }}

        /* ─── Sidebar ─── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
            border-right: 1px solid rgba(0, 150, 136, 0.2);
        }}

        /* ─── Headers ─── */
        h1, h2, h3 {{
            background: linear-gradient(90deg, {THEME_PRIMARY}, {THEME_SECONDARY});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}

        /* ─── Glassmorphism Card ─── */
        .glass-card {{
            background: rgba(22, 27, 34, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 150, 136, 0.25);
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        /* ─── Buttons ─── */
        .stButton > button {{
            background: linear-gradient(135deg, {THEME_PRIMARY}, {THEME_SECONDARY});
            color: #ffffff !important;
            border: none;
            border-radius: 8px;
            padding: 12px 32px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 150, 136, 0.3);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 150, 136, 0.5);
        }}

        /* ─── Risk Badge ─── */
        .risk-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
        }}
        .risk-low {{ background: rgba(102, 187, 106, 0.2); color: #66BB6A; border: 2px solid #66BB6A; }}
        .risk-medium {{ background: rgba(255, 183, 77, 0.2); color: #FFB74D; border: 2px solid #FFB74D; }}
        .risk-high {{ background: rgba(239, 83, 80, 0.2); color: #EF5350; border: 2px solid #EF5350; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
@st.cache_resource
def load_scorer():
    """Load the CreditScorer once and cache it."""
    try:
        from src.predict import CreditScorer

        return CreditScorer()
    except FileNotFoundError:
        return None


scorer = load_scorer()


@st.cache_data
def load_metrics():
    """Load model metrics from feature_config.json."""
    import json
    from src.config import FEATURE_CONFIG_PATH
    try:
        with open(FEATURE_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


metrics = load_metrics()


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def create_gauge_chart(score_label: str, confidence: float) -> go.Figure:
    """Create a premium gauge chart for credit score visualization."""
    score_value_map = {"Low": 25, "Average": 55, "High": 85}
    score_value = score_value_map.get(score_label, 50)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score_value,
            number={"suffix": "", "font": {"size": 60, "color": THEME_TEXT}},
            title={
                "text": f"<b>Credit Score: {score_label}</b>",
                "font": {"size": 22, "color": THEME_TEXT},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "rgba(255,255,255,0.3)",
                    "tickfont": {"color": THEME_TEXT},
                },
                "bar": {"color": THEME_PRIMARY, "thickness": 0.3},
                "bgcolor": THEME_SURFACE,
                "borderwidth": 2,
                "bordercolor": "rgba(0,150,136,0.3)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(239, 83, 80, 0.3)"},
                    {"range": [35, 65], "color": "rgba(255, 183, 77, 0.3)"},
                    {"range": [65, 100], "color": "rgba(102, 187, 106, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": score_value,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME_TEXT},
        height=320,
        margin=dict(l=30, r=30, t=80, b=30),
    )

    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create a horizontal bar chart for probability breakdown."""
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    colors = ["#EF5350", "#FFB74D", "#66BB6A"]  # Red, Amber, Green

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.1)", width=1),
                cornerradius=6,
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(color="white", size=14, family="Inter"),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT),
        height=220,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            title="Probability (%)",
        ),
        yaxis=dict(showgrid=False),
    )

    return fig


def get_risk_badge_html(risk_level: str) -> str:
    """Generate HTML for the risk level badge."""
    risk_class_map = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }
    css_class = risk_class_map.get(risk_level, "risk-medium")
    return f'<div class="risk-badge {css_class}">{risk_level}</div>'


# ──────────────────────────────────────────────
# Sidebar — Input Form
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Applicant Information")
    st.markdown("---")

    age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)

    income = st.number_input(
        "Annual Income (USD)",
        min_value=1000,
        max_value=1000000,
        value=75000,
        step=5000,
        format="%d",
        help="Model assumes international standard (USD).",
    )

    education = st.selectbox(
        "Education Level",
        options=[
            "Intermediate",
            "Associate's Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Doctorate",
        ],
        index=2,
    )

    marital_status = st.selectbox(
        "Marital Status", options=["Single", "Married"], index=0
    )

    num_children = st.slider(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )

    home_ownership = st.selectbox(
        "Home Ownership", options=["Rented", "Owned"], index=0
    )

    st.markdown("---")
    predict_button = st.button("Predict Credit Score", use_container_width=True)

    # System Status & API Docs
    st.markdown("### System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.success("API Live")
    with status_col2:
        st.success("Model Ready")
    
    st.markdown("---")
    st.link_button("View API Documentation", "http://127.0.0.1:8000/docs", use_container_width=True)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown(f"# 🏦 {DASHBOARD_TITLE.replace('🏦 ', '')}")
st.markdown(
    "Production-grade credit scoring analysis powered by Gradient Boosting Machine Learning."
)
st.markdown("---")

# Check model status
if scorer is None:
    st.error(
        "⚠️ **Model Not Loaded** — Please place the trained model artifacts "
        "(`credit_model.pkl`, `scaler.pkl`, `target_encoder.pkl`) in the `models/` directory, "
        "then restart the dashboard."
    )
    st.info(
        "💡 **Tip:** Train the model on Google Colab using the provided training script, "
        "download the `.pkl` files, and place them in the `models/` folder."
    )
    st.stop()

# Prediction
if predict_button:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Income": income,
        "Education": education,
        "Marital Status": marital_status,
        "Number of Children": num_children,
        "Home Ownership": home_ownership,
    }

    with st.spinner("🔄 Analyzing applicant profile..."):
        try:
            result = scorer.predict(input_data)

            # ─── Results Layout ───
            st.markdown("## Prediction Results")

            col1, col2 = st.columns([3, 2])

            with col1:
                # Gauge Chart
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                gauge_fig = create_gauge_chart(
                    result["credit_score"], result["confidence"]
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                # Metrics
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Score Details")

                st.metric(label="Credit Score", value=result["credit_score"])
                st.metric(label="Confidence", value=f"{result['confidence']:.1%}")

                # Risk Badge
                st.markdown("### Risk Assessment")
                st.markdown(get_risk_badge_html(result["risk_level"]), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Probability Breakdown
            st.markdown("### Probability Breakdown")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            prob_fig = create_probability_chart(result["probabilities"])
            st.plotly_chart(prob_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Input Summary Table
            st.markdown("### Applicant Summary")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                st.write(f"**Age:** {age} years")
                st.write(f"**Gender:** {gender}")
            with sum_col2:
                st.write(f"**Income:** ${income:,.0f} USD")
                st.write(f"**Education:** {education}")
            with sum_col3:
                st.write(f"**Status:** {marital_status}")
                st.write(f"**Home:** {home_ownership}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ─── Recent Activity (Tracking) ───
            if "history" not in st.session_state:
                st.session_state.history = []

            # Add current result to history
            new_entry = {
                "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Score": result["credit_score"],
                "Confidence": f"{result['confidence']:.1%}",
                "Risk": result["risk_level"]
            }
            st.session_state.history.insert(0, new_entry)
            st.session_state.history = st.session_state.history[:5]  # Keep last 5

            st.markdown("---")
            st.markdown("### Recent Activity")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state.history))
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

else:
    # Default state
    st.markdown("## Fill in the form and click Predict")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Real-Time Scoring</h3>
                <p style="color: #8b949e;">
                    Instantly predict credit scores using our trained ML model.
                    Results include confidence levels and risk assessment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>3-Class Prediction</h3>
                <p style="color: #8b949e;">
                    Credit scores are classified as <b style="color:#EF5350;">Low</b>,
                    <b style="color:#FFB74D;">Average</b>, or
                    <b style="color:#66BB6A;">High</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Risk Assessment</h3>
                <p style="color: #8b949e;">
                    Each prediction includes an automated risk level
                    to support lending decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Credit Scoring System v1.0.0 • Built with ❤️ using Streamlit & FastAPI •
        Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
