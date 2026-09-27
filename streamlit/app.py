
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from queries import (  # noqa: E402
    get_ai_predictions,
    get_ticket_analytics,
)

from src.aiops.micross.risk_predictor import (  # noqa: E402
    MicroSSRiskPredictor,
)

from src.rag.rag_engine import (  # noqa: E402
    retrieve_similar_tickets,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "tickets_clean.csv"
)


FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

FASTAPI_API_KEY = os.getenv("API_SECRET_KEY", "").strip()

AGENT_URL = os.getenv(
    "AGENT_URL",
    "http://127.0.0.1:8001",
).rstrip("/")


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HARDTEC IT Incident Detection, Prediction & Intelligent Support",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .risk-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #d1d5db;
        background-color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .recommendation-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #d1d5db;
        background-color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .rag-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #d1d5db;
        background-color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎫 HARDTEC IT Incident Detection, Prediction &amp; Intelligent Support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered IT ticket classification, incident prediction,
        intelligent support and operational analytics.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FASTAPI CLIENT
# ============================================================

def predict_via_api(ticket_text: str):
    """
    Send a ticket to the FastAPI backend.

    Architecture:

        Streamlit
            ↓ HTTP POST
        FastAPI /predict
            ↓
        ML ticket pipeline
            ↓
        Type → Queue → Priority
    """

    if not isinstance(ticket_text, str):
        raise ValueError(
            "Ticket description must be a string."
        )

    ticket_text = ticket_text.strip()

    if not ticket_text:
        raise ValueError(
            "Ticket description cannot be empty."
        )

    try:

        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json={
                "ticket_text": ticket_text,
            },
            headers=(
                {"X-API-Key": FASTAPI_API_KEY}
                if FASTAPI_API_KEY
                else {}
            ),
            timeout=30,
        )

    except requests.exceptions.ConnectionError as e:

        raise RuntimeError(
            "Unable to connect to the FastAPI backend. "
            f"Please make sure FastAPI is running at "
            f"{FASTAPI_URL}."
        ) from e

    except requests.exceptions.Timeout as e:

        raise RuntimeError(
            "The FastAPI backend did not respond "
            "within the expected time."
        ) from e

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"FastAPI request failed: {e}"
        ) from e

    if response.status_code != 200:

        try:
            error_data = response.json()

        except ValueError:
            error_data = response.text

        raise RuntimeError(
            f"FastAPI returned HTTP {response.status_code}: "
            f"{error_data}"
        )

    try:

        result = response.json()

    except ValueError as e:

        raise RuntimeError(
            "FastAPI returned an invalid JSON response."
        ) from e

    required_fields = [
        "ticket_text",
        "type",
        "priority",
        "queue",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing_fields:

        raise RuntimeError(
            "Invalid FastAPI response. Missing fields: "
            + ", ".join(missing_fields)
        )

    return result


def analyze_with_agent(ticket_text: str, top_k: int, allow_external_llm: bool):
    """Ask the agent service to orchestrate prediction and support actions."""
    response = requests.post(
        f"{AGENT_URL}/agent/analyze",
        json={
            "ticket_text": ticket_text.strip(),
            "top_k": top_k,
            "allow_external_llm": allow_external_llm,
        },
        headers=(
            {"X-API-Key": FASTAPI_API_KEY}
            if FASTAPI_API_KEY
            else {}
        ),
        timeout=45,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Agent returned HTTP {response.status_code}: {detail}")

    return response.json()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=300)
def load_analytics():
    """
    Load ticket analytics from the local DuckDB backend.
    """

    try:

        df = get_ticket_analytics()

        if df is None:
            return pd.DataFrame()

        return df

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_predictions():
    """
    Load local AI prediction history.
    """

    try:

        df = get_ai_predictions()

        if df is None:
            return pd.DataFrame()

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# OPERATIONAL RISK ASSESSMENT
# ============================================================

def calculate_risk_assessment(
    predicted_type: str,
    predicted_priority: str,
    predicted_queue: str,
):
    """
    Transparent operational risk assessment.

    IMPORTANT:
    This is NOT the AIOps forecasting model.

    It is only an operational rule-based layer
    for the ticket classification workflow.
    """

    score = 0

    priority = str(
        predicted_priority
    ).upper()

    ticket_type = str(
        predicted_type
    ).strip()

    queue = str(
        predicted_queue
    ).strip()

    if priority == "HIGH":
        score += 70

    elif priority == "MEDIUM":
        score += 45

    else:
        score += 20

    if ticket_type.lower() == "incident":
        score += 15

    elif ticket_type.lower() == "problem":
        score += 10

    elif ticket_type.lower() == "change":
        score += 5

    if queue.lower() == "service outages and maintenance":
        score += 15

    score = min(score, 100)

    if score >= 80:
        risk_level = "CRITICAL"

    elif score >= 60:
        risk_level = "HIGH"

    elif score >= 35:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    escalation_required = score >= 70

    return {
        "score": score,
        "level": risk_level,
        "escalation": escalation_required,
    }


# ============================================================
# SUPPORT RECOMMENDATION
# ============================================================

def get_support_recommendation(
    predicted_type: str,
    predicted_priority: str,
    predicted_queue: str,
):
    """
    Generate an operational recommendation
    based on the ML predictions.
    """

    priority = str(
        predicted_priority
    ).upper()

    queue = str(
        predicted_queue
    )

    if priority == "HIGH":

        return (
            "Immediate attention recommended. "
            f"Route the ticket to **{queue}** and "
            "prioritize investigation. "
            "Consider escalation to the responsible "
            "support team."
        )

    if priority == "MEDIUM":

        return (
            f"Assign the ticket to **{queue}**. "
            "Monitor the resolution process and "
            "ensure the issue is handled within the "
            "expected service level."
        )

    return (
        f"Route the ticket to **{queue}** for standard "
        "processing. Resolution can follow the normal "
        "support workflow."
    )


# ============================================================
# AIOPS PREDICTOR
# ============================================================

@st.cache_resource
def load_aiops_predictor():
    """
    Load the MicroSS XGBoost V3.1 predictor.

    Architecture:

        MicroSS metrics
              ↓
        5-minute windows
              ↓
        115 base metrics
              ↓
        Temporal features
              ↓
        1,380 features
              ↓
        XGBoost V3.1
              ↓
        Incident risk next 15 min
    """

    return MicroSSRiskPredictor()


@st.cache_data(ttl=300)
def load_aiops_overview():
    """Build a compact recent AIOps incident-risk overview for the home page."""
    predictor = load_aiops_predictor()
    timestamps = predictor.get_available_timestamps()

    if not timestamps:
        return predictor.get_data_info(), pd.DataFrame()

    recent_timestamps = timestamps[-24:]
    rows = [predictor.predict(timestamp) for timestamp in recent_timestamps]
    risk_df = pd.DataFrame(rows)
    risk_df["timestamp"] = pd.to_datetime(risk_df["timestamp"])
    return predictor.get_data_info(), risk_df


def render_aiops_overview():
    """Render the AIOps command center shown on the main dashboard."""
    st.markdown("---")
    st.subheader("AIOps incident command center")
    st.caption(
        "Recent MicroSS signals and the predicted incident risk for the next 15 minutes."
    )

    info, risk_df = load_aiops_overview()
    if risk_df.empty:
        st.warning("No recent AIOps observations are available.")
        return

    latest = risk_df.iloc[-1]
    alert_count = int(risk_df["alert"].sum())
    average_risk = float(risk_df["risk_percentage"].mean())

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric(
        "Current risk",
        f"{latest['risk_percentage']:.1f}%",
        f"{latest['risk_level']} level",
    )
    metric_col2.metric("Current alert", "YES" if latest["alert"] else "NO")
    metric_col3.metric("Alerts in window", alert_count)
    metric_col4.metric("Average risk", f"{average_risk:.1f}%")
    metric_col5.metric("Model", info["model_version"])

    chart_df = risk_df[["timestamp", "risk_percentage"]].rename(
        columns={"timestamp": "Timestamp", "risk_percentage": "Risk (%)"}
    ).set_index("Timestamp")
    st.line_chart(chart_df, height=260)

    alert_df = risk_df[risk_df["alert"]].copy().sort_values(
        "timestamp", ascending=False
    )
    if alert_df.empty:
        st.success("No incident alert in the latest AIOps observation window.")
    else:
        st.warning(f"{len(alert_df)} incident signal(s) require operational review.")
        st.dataframe(
            alert_df[
                [
                    "timestamp",
                    "risk_percentage",
                    "risk_level",
                    "threshold_percentage",
                ]
            ].rename(
                columns={
                    "timestamp": "Observation",
                    "risk_percentage": "Risk (%)",
                    "risk_level": "Level",
                    "threshold_percentage": "Threshold (%)",
                }
            ),
            hide_index=True,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "AI Ticket Analyzer",
        "Support Agent",
        "Incident Prediction",
        "Intelligent Support",
        "Analytics",
    ],
)


# ============================================================
# BACKEND STATUS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Backend")

try:

    health_response = requests.get(
        f"{FASTAPI_URL}/health",
        timeout=3,
    )

    if health_response.status_code == 200:

        health_data = health_response.json()

        st.sidebar.success(
            "FastAPI: Online"
        )

        st.sidebar.caption(
            f"ML Pipeline: "
            f"{health_data.get('ml_pipeline', 'unknown')}"
        )

        backend_status = health_data.get(
            "snowflake",
            "unknown",
        )

        if str(backend_status).lower() in [
            "online",
            "connected",
            "available",
        ]:

            st.sidebar.success(
                f"Database: {backend_status}"
            )

        else:

            st.sidebar.warning(
                f"Database: {backend_status}"
            )

    else:

        st.sidebar.warning(
            "FastAPI: Unavailable"
        )

except requests.exceptions.RequestException:

    st.sidebar.error(
        "FastAPI: Offline"
    )

st.sidebar.caption(
    f"API: {FASTAPI_URL}"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">📊 Dashboard</div>',
        unsafe_allow_html=True,
    )

    analytics_df = load_analytics()

    try:
        render_aiops_overview()
    except Exception as e:
        st.error("Unable to load the AIOps incident overview.")
        st.caption(str(e))

    if analytics_df.empty:

        st.warning(
            "Local ticket analytics are currently unavailable."
        )

        st.info(
            """
            The ticket analytics module is running with the
            local DuckDB backend.

            The AIOps Incident Prediction module continues
            to operate locally using the MicroSS dataset.
            """
        )

    else:

        st.sidebar.markdown("---")

        st.sidebar.subheader(
            "Dashboard Filters"
        )

        type_values = sorted(
            analytics_df["TICKET_TYPE"]
            .dropna()
            .unique()
            .tolist()
        )

        priority_values = sorted(
            analytics_df["PRIORITY"]
            .dropna()
            .unique()
            .tolist()
        )

        queue_values = sorted(
            analytics_df["QUEUE"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_types = st.sidebar.multiselect(
            "Ticket Type",
            type_values,
            default=type_values,
        )

        selected_priorities = st.sidebar.multiselect(
            "Priority",
            priority_values,
            default=priority_values,
        )

        selected_queues = st.sidebar.multiselect(
            "Support Queue",
            queue_values,
            default=queue_values,
        )

        filtered_df = analytics_df[
            analytics_df["TICKET_TYPE"].isin(
                selected_types
            )
            & analytics_df["PRIORITY"].isin(
                selected_priorities
            )
            & analytics_df["QUEUE"].isin(
                selected_queues
            )
        ].copy()

        total_tickets = int(
            filtered_df[
                "TOTAL_TICKETS"
            ].sum()
        )

        incidents = int(
            filtered_df.loc[
                filtered_df["TICKET_TYPE"]
                .astype(str)
                .str.lower()
                == "incident",
                "TOTAL_TICKETS",
            ].sum()
        )

        high_priority = int(
            filtered_df[
                "HIGH_PRIORITY_TICKETS"
            ].sum()
        )

        medium_priority = int(
            filtered_df[
                "MEDIUM_PRIORITY_TICKETS"
            ].sum()
        )

        low_priority = int(
            filtered_df[
                "LOW_PRIORITY_TICKETS"
            ].sum()
        )

        col1, col2, col3, col4, col5 = st.columns(
            5
        )

        col1.metric(
            "Total Tickets",
            f"{total_tickets:,}",
        )

        col2.metric(
            "Incidents",
            f"{incidents:,}",
        )

        col3.metric(
            "High",
            f"{high_priority:,}",
        )

        col4.metric(
            "Medium",
            f"{medium_priority:,}",
        )

        col5.metric(
            "Low",
            f"{low_priority:,}",
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            type_chart = (
                filtered_df
                .groupby("TICKET_TYPE")[
                    "TOTAL_TICKETS"
                ]
                .sum()
                .reset_index()
            )

            fig_type = px.bar(
                type_chart,
                x="TICKET_TYPE",
                y="TOTAL_TICKETS",
                title="Tickets by Type",
            )

            st.plotly_chart(
                fig_type,
                use_container_width=True,
            )

        with col2:

            priority_chart = (
                filtered_df
                .groupby("PRIORITY")[
                    "TOTAL_TICKETS"
                ]
                .sum()
                .reset_index()
            )

            fig_priority = px.pie(
                priority_chart,
                names="PRIORITY",
                values="TOTAL_TICKETS",
                title="Tickets by Priority",
            )

            st.plotly_chart(
                fig_priority,
                use_container_width=True,
            )

        queue_chart = (
            filtered_df
            .groupby("QUEUE")[
                "TOTAL_TICKETS"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "TOTAL_TICKETS",
                ascending=False,
            )
        )

        fig_queue = px.bar(
            queue_chart,
            x="QUEUE",
            y="TOTAL_TICKETS",
            title="Tickets by Support Queue",
        )

        st.plotly_chart(
            fig_queue,
            use_container_width=True,
        )

        st.subheader(
            "Analytics Data"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AI TICKET ANALYZER
# ============================================================

elif page == "AI Ticket Analyzer":

    st.markdown(
        '<div class="section-title">🤖 AI Ticket Analyzer</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Enter an IT support ticket description.

        The AI pipeline predicts:

        **Ticket Type → Support Queue → Priority**
        """
    )

    ticket_text = st.text_area(
        "Ticket Description",
        placeholder=(
            "Example: "
            "Users are unable to access the company VPN "
            "since this morning."
        ),
        height=180,
    )

    analyze_button = st.button(
        "🔍 Analyze Ticket",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        if not ticket_text.strip():

            st.warning(
                "Please enter a ticket description."
            )

        else:

            try:

                result = predict_via_api(
                    ticket_text
                )

                predicted_type = result[
                    "type"
                ]

                predicted_priority = result[
                    "priority"
                ]

                predicted_queue = result[
                    "queue"
                ]

                st.success(
                    "Prediction successfully processed by FastAPI."
                )

                st.markdown("---")

                st.subheader(
                    "Prediction Results"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Ticket Type",
                    predicted_type,
                )

                col2.metric(
                    "Support Queue",
                    predicted_queue,
                )

                col3.metric(
                    "Priority",
                    predicted_priority,
                )

                risk = calculate_risk_assessment(
                    predicted_type,
                    predicted_priority,
                    predicted_queue,
                )

                st.subheader(
                    "Operational Risk"
                )

                risk_col1, risk_col2, risk_col3 = st.columns(
                    3
                )

                risk_col1.metric(
                    "Risk Score",
                    f"{risk['score']}/100",
                )

                risk_col2.metric(
                    "Risk Level",
                    risk["level"],
                )

                risk_col3.metric(
                    "Escalation",
                    "Required"
                    if risk["escalation"]
                    else "Not Required",
                )

                st.info(
                    """
                    This operational risk score is a
                    transparent rule-based layer based on
                    ticket classification. It is not the
                    AIOps incident forecasting model.
                    """
                )

                st.subheader(
                    "Recommended Action"
                )

                recommendation = (
                    get_support_recommendation(
                        predicted_type,
                        predicted_priority,
                        predicted_queue,
                    )
                )

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader(
                    "Prediction Details"
                )

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:

                    st.write(
                        "**Ticket Type:**",
                        predicted_type,
                    )

                    st.write(
                        "**Support Queue:**",
                        predicted_queue,
                    )

                    st.write(
                        "**Priority:**",
                        predicted_priority,
                    )

                with detail_col2:

                    st.write(
                        "**Backend:**",
                        "FastAPI",
                    )

                    st.write(
                        "**Persistence:**",
                        "DuckDB + local CSV",
                    )

                    if "confidence" in result:

                        st.write(
                            "**API Confidence:**",
                            result["confidence"],
                        )

                        st.caption(
                            "The current API confidence value "
                            "is a placeholder and is not a "
                            "calibrated ML probability."
                        )

            except Exception as e:

                st.error(
                    "The AI ticket analysis failed."
                )

                st.exception(e)


# ============================================================
# SUPPORT AGENT
# ============================================================

elif page == "Support Agent":

    st.markdown(
        '<div class="section-title">🧠 Intelligent Support Agent</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "The agent combines ticket classification, historical context and an actionable support plan."
    )

    with st.form("agent_analysis_form"):
        agent_text = st.text_area(
            "Incident description",
            placeholder="Example: Production users cannot connect to the VPN since 09:15.",
            height=160,
        )
        agent_top_k = st.slider("Historical cases", min_value=1, max_value=10, value=3)
        allow_external_llm = st.checkbox(
            "Use external LLM summary",
            value=False,
            help="Requires AGENT_LLM_URL and AGENT_LLM_API_KEY. Local analysis works without them.",
        )
        agent_submit = st.form_submit_button("Analyze with the agent", type="primary")

    if agent_submit:
        if not agent_text.strip():
            st.warning("Please enter an incident description.")
        else:
            try:
                with st.spinner("The support agent is coordinating the analysis..."):
                    agent_result = analyze_with_agent(
                        agent_text,
                        agent_top_k,
                        allow_external_llm,
                    )

                prediction = agent_result.get("prediction", {})
                result_col1, result_col2, result_col3 = st.columns(3)
                result_col1.metric("Type", prediction.get("ticket_type", "Unknown"))
                result_col2.metric("Priority", prediction.get("priority", "Unknown"))
                result_col3.metric("Queue", prediction.get("queue", "Unknown"))

                st.subheader("Agent recommendation")
                st.info(agent_result.get("summary", "No summary returned."))

                st.subheader("Recommended actions")
                for index, action in enumerate(
                    agent_result.get("recommended_actions", []),
                    start=1,
                ):
                    st.write(f"{index}. {action}")

                similar_tickets = agent_result.get("similar_tickets", [])
                st.subheader("Historical context")
                if not similar_tickets:
                    st.caption("No similar historical ticket was found.")
                else:
                    for index, ticket in enumerate(similar_tickets, start=1):
                        score = float(ticket.get("score", 0.0)) * 100
                        with st.expander(f"Case #{index} - {score:.1f}% similarity"):
                            st.write(ticket.get("document", ""))

                if agent_result.get("human_approval_required", True):
                    st.warning(
                        "Human approval is required before any irreversible operational action."
                    )

            except requests.exceptions.RequestException as exc:
                st.error(f"Unable to reach the agent service at {AGENT_URL}.")
                st.caption(str(exc))
            except Exception as exc:
                st.error("The support agent analysis failed.")
                st.caption(str(exc))


# ============================================================
# AIOPS INCIDENT PREDICTION
# ============================================================

elif page == "Incident Prediction":

    st.markdown(
        '<div class="section-title">🚨 AIOps Incident Prediction</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Predict the risk of an operational incident during
        the next **15 minutes** using infrastructure and
        service metrics from the MicroSS dataset.
        """
    )

    st.info(
        """
        Model: **XGBoost V3.1**

        115 base metrics → temporal features →
        1,380 features → incident risk prediction.
        """
    )

    try:

        predictor = load_aiops_predictor()

        info = predictor.get_data_info()

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "Model Information"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Model",
            info["model_version"],
        )

        col2.metric(
            "Base Metrics",
            info["base_metrics"],
        )

        col3.metric(
            "Features",
            info["features"],
        )

        col4.metric(
            "Horizon",
            "15 min",
        )

        st.markdown("---")

        # ----------------------------------------------------
        # TIMESTAMP SELECTION
        # ----------------------------------------------------

        timestamps = predictor.get_available_timestamps()

        if not timestamps:

            st.error(
                "No valid timestamps are available."
            )

        else:

            timestamp_options = timestamps[
                -500:
            ]

            selected_timestamp = st.selectbox(
                "Select observation timestamp",
                timestamp_options,
                index=len(timestamp_options) - 1,
                format_func=lambda x: x.strftime(
                    "%Y-%m-%d %H:%M"
                ),
            )

            predict_button = st.button(
                "🔮 Predict Incident Risk",
                type="primary",
                use_container_width=True,
            )

            if predict_button:

                result = predictor.predict(
                    selected_timestamp
                )

                st.markdown("---")

                # ------------------------------------------------
                # RISK RESULT
                # ------------------------------------------------

                st.subheader(
                    "Incident Risk"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Risk Score",
                    f"{result['risk_percentage']:.2f}%",
                )

                col2.metric(
                    "Threshold",
                    f"{result['threshold_percentage']:.0f}%",
                )

                col3.metric(
                    "Risk Level",
                    result["risk_level"],
                )

                col4.metric(
                    "Alert",
                    "YES"
                    if result["alert"]
                    else "NO",
                )

                # ------------------------------------------------
                # ALERT
                # ------------------------------------------------

                if result["alert"]:

                    st.error(
                        "🚨 HIGH RISK: The model predicts "
                        "an elevated probability of an incident "
                        "during the next 15 minutes."
                    )

                else:

                    st.success(
                        "✅ No high-risk incident alert "
                        "for this observation."
                    )

                # ------------------------------------------------
                # MODEL EXPLANATION
                # ------------------------------------------------

                st.info(
                    f"""
                    Prediction timestamp: {result['timestamp']}

                    Target: {result['target']}

                    Model version: {result['model_version']}

                    Alert threshold: 25%

                    The prediction is based on historical
                    infrastructure and service metrics, including
                    temporal lags, variations and rolling statistics.
                    """
                )

                # ------------------------------------------------
                # RECENT METRICS
                # ------------------------------------------------

                st.subheader(
                    "Recent Metric Evolution"
                )

                recent_metrics = predictor.get_recent_metrics(
                    selected_timestamp,
                    minutes=15,
                )

                if not recent_metrics.empty:

                    metric_columns = (
                        predictor.get_base_metrics()
                    )

                    available_metric_columns = [
                        column
                        for column in metric_columns
                        if column in recent_metrics.columns
                    ]

                    display_metrics = (
                        available_metric_columns[:10]
                    )

                    if display_metrics:

                        chart_df = (
                            recent_metrics[
                                display_metrics
                            ]
                            .reset_index()
                            .melt(
                                id_vars="timestamp",
                                var_name="metric",
                                value_name="value",
                            )
                        )

                        fig = px.line(
                            chart_df,
                            x="timestamp",
                            y="value",
                            color="metric",
                            title="Recent MicroSS Metrics",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                # ------------------------------------------------
                # METHODOLOGICAL NOTE
                # ------------------------------------------------

                st.warning(
                    """
                    Methodological note:

                    The V3.1 model introduces temporal features
                    such as 5/10/15-minute lags, deltas,
                    percentage changes and 15-minute rolling
                    statistics.

                    The test ROC-AUC is approximately 0.51,
                    therefore this model should be presented as
                    an AIOps predictive baseline rather than as
                    a highly accurate production forecasting model.
                    """
                )

    except Exception as e:

        st.error(
            "Unable to load the AIOps Incident Prediction model."
        )

        st.exception(e)


# ============================================================
# INTELLIGENT SUPPORT
# ============================================================

elif page == "Intelligent Support":

    st.markdown(
        '<div class="section-title">🧠 Intelligent Support</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Combine AI ticket classification with semantic
        retrieval of historical support cases to help
        support agents resolve incidents faster.
        """
    )

    st.info(
        """
        **RAG Pipeline**

        Ticket → Sentence Transformer → FAISS →
        Similar Historical Tickets → Historical Solutions
        """
    )

    support_text = st.text_area(
        "Ticket Description",
        placeholder=(
            "Example: "
            "The data analytics platform crashed because "
            "of insufficient RAM. We restarted the server "
            "but the problem persists."
        ),
        height=180,
        key="support_text",
    )

    support_button = st.button(
        "🧠 Analyze & Find Similar Tickets",
        type="primary",
        use_container_width=True,
    )

    if support_button:

        if not support_text.strip():

            st.warning(
                "Please enter a ticket description."
            )

        else:

            try:

                # ==================================================
                # STEP 1 — AI CLASSIFICATION
                # ==================================================

                with st.spinner(
                    "Analyzing ticket with the AI pipeline..."
                ):

                    result = predict_via_api(
                        support_text
                    )

                predicted_type = result[
                    "type"
                ]

                predicted_priority = result[
                    "priority"
                ]

                predicted_queue = result[
                    "queue"
                ]

                st.subheader(
                    "🤖 AI Classification"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Type",
                    predicted_type,
                )

                col2.metric(
                    "Priority",
                    predicted_priority,
                )

                col3.metric(
                    "Queue",
                    predicted_queue,
                )

                # ==================================================
                # STEP 2 — OPERATIONAL RECOMMENDATION
                # ==================================================

                st.subheader(
                    "🎯 Recommended Action"
                )

                recommendation = (
                    get_support_recommendation(
                        predicted_type,
                        predicted_priority,
                        predicted_queue,
                    )
                )

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ==================================================
                # STEP 3 — RAG RETRIEVAL
                # ==================================================

                st.subheader(
                    "🔎 Similar Historical Tickets"
                )

                st.caption(
                    "Semantic search powered by "
                    "Sentence Transformer embeddings + FAISS."
                )

                with st.spinner(
                    "Searching the RAG knowledge base..."
                ):

                    rag_results = (
                        retrieve_similar_tickets(
                            support_text,
                            top_k=3,
                        )
                    )

                if not rag_results:

                    st.info(
                        "No similar historical tickets found."
                    )

                else:

                    for rank, item in enumerate(
                        rag_results,
                        start=1,
                    ):

                        similarity = (
                            float(item["score"]) * 100
                        )

                        document = item["document"]

                        with st.expander(
                            f"Match #{rank} — "
                            f"{similarity:.1f}% semantic similarity"
                        ):

                            st.markdown(
                                """
                                <div class="rag-card">
                                """,
                                unsafe_allow_html=True,
                            )

                            st.write(
                                "**Historical Case**"
                            )

                            st.markdown(
                                document
                            )

                            st.markdown(
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            st.caption(
                                f"FAISS semantic similarity: "
                                f"{similarity:.2f}%"
                            )

                    # ==================================================
                    # STEP 4 — BEST HISTORICAL SOLUTION
                    # ==================================================

                    best_match = rag_results[0]

                    best_document = (
                        best_match["document"]
                    )

                    st.subheader(
                        "💡 Historical Solution"
                    )

                    if "Historical solution:" in best_document:

                        historical_solution = (
                            best_document.split(
                                "Historical solution:",
                                1,
                            )[1]
                            .strip()
                        )

                        if historical_solution:

                            st.info(
                                historical_solution
                            )

                        else:

                            st.info(
                                "The best matching historical "
                                "case does not contain a solution."
                            )

                    else:

                        st.info(
                            "The best matching historical "
                            "case does not contain a solution."
                        )

                    st.caption(
                        f"Best semantic match: "
                        f"{best_match['score'] * 100:.2f}%"
                    )

                # ==================================================
                # STEP 5 — HUMAN VALIDATION
                # ==================================================

                st.warning(
                    """
                    Human validation required:

                    RAG recommendations are retrieved from
                    historical support cases. Semantic similarity
                    is not accuracy, and historical solutions should
                    be validated by a support agent before being
                    applied to the current incident.
                    """
                )

            except Exception as e:

                st.error(
                    "Intelligent Support analysis failed."
                )

                st.exception(e)


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="section-title">📈 Analytics</div>',
        unsafe_allow_html=True,
    )

    analytics_df = load_analytics()

    if analytics_df.empty:

        st.warning(
            "Local analytics are currently unavailable."
        )

        st.info(
            """
            Ticket analytics use the local DuckDB backend.

            This does not affect the AIOps prediction engine,
            FastAPI ticket classification or RAG-based
            Intelligent Support.
            """
        )

    else:

        st.subheader(
            "General Analytics"
        )

        st.dataframe(
            analytics_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # TYPE DISTRIBUTION
        # ----------------------------------------------------

        type_chart = (
            analytics_df
            .groupby("TICKET_TYPE")[
                "TOTAL_TICKETS"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "TOTAL_TICKETS",
                ascending=False,
            )
        )

        fig_type = px.bar(
            type_chart,
            x="TICKET_TYPE",
            y="TOTAL_TICKETS",
            title="Ticket Type Distribution",
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # QUEUE DISTRIBUTION
        # ----------------------------------------------------

        queue_chart = (
            analytics_df
            .groupby("QUEUE")[
                "TOTAL_TICKETS"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "TOTAL_TICKETS",
                ascending=False,
            )
        )

        fig_queue = px.bar(
            queue_chart,
            x="QUEUE",
            y="TOTAL_TICKETS",
            title="Support Queue Distribution",
        )

        st.plotly_chart(
            fig_queue,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # PRIORITY DISTRIBUTION
        # ----------------------------------------------------

        priority_chart = (
            analytics_df
            .groupby("PRIORITY")[
                "TOTAL_TICKETS"
            ]
            .sum()
            .reset_index()
        )

        fig_priority = px.pie(
            priority_chart,
            names="PRIORITY",
            values="TOTAL_TICKETS",
            title="Priority Distribution",
        )

        st.plotly_chart(
            fig_priority,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # TYPE × PRIORITY
        # ----------------------------------------------------

        type_priority = (
            analytics_df
            .groupby(
                [
                    "TICKET_TYPE",
                    "PRIORITY",
                ]
            )[
                "TOTAL_TICKETS"
            ]
            .sum()
            .reset_index()
        )

        fig_type_priority = px.bar(
            type_priority,
            x="TICKET_TYPE",
            y="TOTAL_TICKETS",
            color="PRIORITY",
            barmode="group",
            title="Ticket Type × Priority",
        )

        st.plotly_chart(
            fig_type_priority,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # AI PREDICTION HISTORY
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Prediction History"
        )

        prediction_df = load_predictions()

        if prediction_df.empty:

            st.info(
                "No AI prediction history available."
            )

        else:

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        HARDTEC Intelligent Ticketing |
        AIOps + AI-powered IT Service Management |
        Data Engineering + Machine Learning + RAG
    </div>
    """,
    unsafe_allow_html=True,
)

