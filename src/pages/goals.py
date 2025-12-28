import streamlit as st
import pandas as pd
import plotly.express as px
from decimal import Decimal
import uuid
from typing import Optional

from src.agents.goal_agent import GoalAgent
from src.core.schemas import AgentRequest, AgentResponse

def _monthly_rate(r_annual: float) -> float:
    return float((Decimal(1) + Decimal(str(r_annual))) ** (Decimal(1) / Decimal(12)) - Decimal(1))


def _future_value_series(
    *,
    current: float,
    monthly: float,
    years: int,
    expected_return_annual: float,
    stepup_annual_pct: float,
) -> pd.DataFrame:
    """Illustrative monthly series using the same compounding/step-up style as quant_engine."""
    n_months = int(years * 12)
    mr = Decimal(str(_monthly_rate(expected_return_annual)))
    step = Decimal(str(stepup_annual_pct))
    bal = Decimal(str(current))
    m = Decimal(str(monthly))

    rows = []
    for i in range(1, n_months + 1):
        bal = bal * (Decimal(1) + mr)
        bal = bal + m
        # step-up at every 12 months (after contribution)
        if i % 12 == 0 and step > 0:
            m = m * (Decimal(1) + step)
        rows.append({"month": i, "value": float(bal)})

    df = pd.DataFrame(rows)
    df["year"] = (df["month"] / 12.0)
    return df


def render():
    st.subheader("Goal planning")

    col_l, col_r = st.columns([0.45, 0.55], gap="large")

    with col_l:
        currency = st.text_input("Currency", value=st.session_state["user_profile"].currency)
        target_amount = st.number_input("Target amount", min_value=0.0, value=10000.0, step=500.0)
        years = st.number_input("Years", min_value=1, max_value=60, value=5, step=1)
        current_savings = st.number_input("Current savings", min_value=0.0, value=0.0, step=100.0)
        monthly_contribution = st.number_input("Monthly contribution", min_value=0.0, value=100.0, step=50.0)

        st.divider()
        risk = st.radio("Risk appetite", options=["low", "medium", "high"], horizontal=True, index=1)
        default_return = {"low": 0.06, "medium": 0.10, "high": 0.14}[risk]
        expected_return_annual = st.number_input("Expected return (annual)", min_value=0.0, max_value=1.0, value=float(default_return), step=0.01)
        inflation_annual = st.number_input("Inflation (annual)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        stepup_annual_pct = st.number_input("Step-up (annual %)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        if st.button("Compute projection", type="primary"):
            goal_payload = {
                "currency": currency,
                "target_amount": str(target_amount),
                "years": str(years),
                "current_savings": str(current_savings),
                "monthly_contribution": str(monthly_contribution),
                "expected_return_annual": str(expected_return_annual),
                "inflation_annual": str(inflation_annual),
                "stepup_annual_pct": str(stepup_annual_pct),
            }
            try:
                req = AgentRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=st.session_state["session_id"],
                    turn_id=int(st.session_state["turn_id"] + 1),
                    user_text="goal projection",
                    user_profile=st.session_state["user_profile"],
                    market_payload={"goal": goal_payload},
                )
                resp = GoalAgent().run(req)
                st.session_state["_goal_resp"] = resp
                st.session_state["_goal_inputs"] = goal_payload
            except Exception as e:
                st.error(f"Goal computation failed: {e}")

    with col_r:
        resp: Optional[AgentResponse] = st.session_state.get("_goal_resp")
        if resp:
            st.markdown(resp.answer_md)
            st.caption(f"Answered by: {resp.agent_name}")

        goal_inputs = st.session_state.get("_goal_inputs")
        if goal_inputs:
            try:
                df = _future_value_series(
                    current=float(goal_inputs["current_savings"]),
                    monthly=float(goal_inputs["monthly_contribution"]),
                    years=int(float(goal_inputs["years"])),
                    expected_return_annual=float(goal_inputs["expected_return_annual"]),
                    stepup_annual_pct=float(goal_inputs["stepup_annual_pct"]),
                )
                fig = px.line(df, x="year", y="value", title="Projection (illustrative, monthly compounding)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("Projection chart unavailable for current inputs.")
