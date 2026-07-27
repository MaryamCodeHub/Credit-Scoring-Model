"""Professional, accessible visual system for the Credit Intelligence Dashboard."""

from __future__ import annotations

from typing import Any


DEEP_NAVY = "#0F172A"
SLATE = "#334155"
BACKGROUND = "#F8FAFC"
SURFACE = "#FFFFFF"
BORDER = "#CBD5E1"
PRIMARY_TEAL = "#0F766E"
SECONDARY_BLUE = "#2563EB"
GOOD = "#15803D"
STANDARD = "#B45309"
POOR = "#B91C1C"
MAIN_TEXT = "#0F172A"
MUTED_TEXT = "#475569"
SYSTEM_FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, "
    "sans-serif"
)

CLASS_COLORS = {
    "Poor": POOR,
    "Standard": STANDARD,
    "Good": GOOD,
}

DASHBOARD_CSS = f"""
<style>
    :root {{
        color-scheme: light;
    }}
    .stApp {{
        background: {BACKGROUND};
        color: {MAIN_TEXT};
        font-family: {SYSTEM_FONT};
    }}
    [data-testid="stSidebar"] {{
        background: {DEEP_NAVY};
        border-right: 1px solid {SLATE};
    }}
    [data-testid="stSidebar"] * {{
        color: #F8FAFC;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
    }}
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        white-space: normal;
        overflow: visible;
    }}
    [data-testid="stForm"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.25rem;
    }}
    .stButton > button,
    .stFormSubmitButton > button {{
        border-radius: 8px;
        min-height: 2.75rem;
        font-weight: 650;
    }}
    h1, h2, h3 {{
        color: {MAIN_TEXT};
        letter-spacing: -0.015em;
    }}
    p, label {{
        color: {MUTED_TEXT};
    }}
    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}
    .block-container {{
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}
    .dashboard-subtitle {{
        color: {MUTED_TEXT};
        font-size: 1.02rem;
        margin-top: -0.6rem;
        margin-bottom: 1.3rem;
    }}
    .section-note {{
        border-left: 4px solid {PRIMARY_TEAL};
        background: {SURFACE};
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        color: {MUTED_TEXT};
        margin: 0.75rem 0 1.25rem;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem 0.8rem 2rem;
        }}
    }}
</style>
"""


def plotly_layout(
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 380,
) -> dict[str, Any]:
    """Return one consistent accessible Plotly layout."""
    return {
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "font": {"family": SYSTEM_FONT, "color": MAIN_TEXT, "size": 13},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "height": height,
        "margin": {"l": 58, "r": 24, "t": 62, "b": 58},
        "xaxis": {
            "title": x_title,
            "gridcolor": "#E2E8F0",
            "zerolinecolor": BORDER,
            "automargin": True,
        },
        "yaxis": {
            "title": y_title,
            "gridcolor": "#E2E8F0",
            "zerolinecolor": BORDER,
            "automargin": True,
        },
        "hoverlabel": {"font": {"family": SYSTEM_FONT}},
        "legend": {"orientation": "h", "y": -0.22},
    }
