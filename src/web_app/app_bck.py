"""Stage 9 — Streamlit UI.

Multi-tab UI:
- Chat: multi-turn context, "Answered by", citations panel, agent trace
- Portfolio: CSV upload, computed metrics, allocation charts
- Market: ticker lookup, trend line, freshness badge
- Goals: projection chart + risk appetite toggle

Run:
  streamlit run src/web_app/app.py
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_agent import GoalAgent
from src.agents.market_agent import MarketAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.core.config import SETTINGS
from src.core.router import Router
from src.core.schemas import AgentRequest, AgentResponse, ChatMessage, UserProfile
from src.utils.logging import setup_logging
from src.utils.market_data import MarketDataService


# -----------------------------
# App setup
# -----------------------------

setup_logging(SETTINGS.log_level)
st.set_page_config(page_title="AI Finance Assistant", layout="wide")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _init_session() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("turn_id", 0)
    st.session_state.setdefault("chat", [])  # list[dict]
    st.session_state.setdefault("last_chat_meta", {})
    st.session_state.setdefault("user_profile", UserProfile())


_init_session()


# -----------------------------
# Helpers (render)
# -----------------------------


def _badge(text: str, kind: str = "info") -> None:
    """Small colored badge using HTML."""
    color = {
        "ok": "#0f9d58",
        "warn": "#f4b400",
        "bad": "#db4437",
        "info": "#4285f4",
    }.get(kind, "#4285f4")
    st.markdown(
        f"""
        <span style="display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;background:{color};color:white;">
          {text}
        </span>
        """,
        unsafe_allow_html=True,
    )


def _render_citations(citations: List[Dict[str, Any]] | List[Any]) -> None:
    if not citations:
        st.caption("No citations")
        return
    rows = []
    for c in citations:
        if hasattr(c, "model_dump"):
            c = c.model_dump()
        c = dict(c)
        rows.append(
            {
                "Title": c.get("title", ""),
                "Score": c.get("score", None),
                "URL": c.get("url", ""),
                "Snippet": c.get("snippet", ""),
            }
        )
    df = pd.DataFrame(rows)
    # Render links via markdown below table (Streamlit DF doesn't render hyperlinks nicely).
    st.dataframe(df[["Title", "Score"]], use_container_width=True, hide_index=True)
    for r in rows:
        title = (r.get("Title") or "(untitled)").strip()
        url = (r.get("URL") or "").strip()
        snippet = (r.get("Snippet") or "").strip()
        if url:
            st.markdown(f"- [{title}]({url})")
        else:
            st.markdown(f"- {title}")
        if snippet:
            st.caption(snippet)


def _render_agent_trace(trace: List[str] | None, router_decision: Optional[Dict[str, Any]] = None) -> None:
    if router_decision:
        st.markdown(
            f"**Route:** `{router_decision.get('intent')}` (conf {router_decision.get('confidence', 0):.2f})"
        )
        if router_decision.get("rationale"):
            st.caption(router_decision["rationale"])
    if not trace:
        st.caption("No trace")
        return
    st.code("\n".join([f"- {t}" for t in trace]), language="text")


def _freshness_badge(*, as_of: Optional[datetime], from_cache: bool, ttl_seconds: Optional[int]) -> Tuple[str, str]:
    if not as_of:
        return ("Freshness: unknown", "info")

    age_s = max(0.0, (_now_utc() - as_of.replace(tzinfo=timezone.utc)).total_seconds())
    age_min = age_s / 60.0

    if ttl_seconds is not None and ttl_seconds > 0:
        stale = age_s > float(ttl_seconds)
    else:
        stale = age_s > 3600.0  # 1h heuristic

    label = f"as_of {as_of.isoformat(timespec='seconds')} · age {age_min:.1f}m"
    if from_cache:
        label += " · cached"

    if stale:
        return ("Stale · " + label, "warn")
    return ("Fresh · " + label, "ok")


# -----------------------------
# Helpers (agents)
# -----------------------------


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    meta: Dict[str, Any]


def _to_chat_messages(turns: List[ChatTurn]) -> List[ChatMessage]:
    msgs: List[ChatMessage] = []
    for t in turns:
        msgs.append(ChatMessage(role=t.role, content=t.content))
    return msgs


def _mk_req(
    *,
    user_text: str,
    user_profile: UserProfile,
    messages: List[ChatMessage],
    extra: Optional[Dict[str, Any]] = None,
) -> AgentRequest:
    extra = extra or {}
    st.session_state["turn_id"] += 1
    return AgentRequest(
        request_id=str(uuid.uuid4()),
        session_id=st.session_state["session_id"],
        turn_id=int(st.session_state["turn_id"]),
        user_text=user_text,
        user_profile=user_profile,
        messages=messages,
        **extra,
    )


def _run_chat_agent(user_text: str) -> Tuple[AgentResponse, Dict[str, Any]]:
    router = Router()
    decision = router.classify(user_text)

    # Build multi-turn context
    turns = [ChatTurn(**t) for t in st.session_state["chat"]]
    msgs = _to_chat_messages(turns[-20:])

    trace: List[str] = ["Router", f"intent={decision.intent}"]

    req_extra: Dict[str, Any] = {"route": decision}

    # Choose agent
    if decision.intent == "MARKET":
        trace.append("MarketAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = MarketAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    if decision.intent == "PORTFOLIO":
        trace.append("PortfolioAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = PortfolioAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    if decision.intent == "GOAL":
        trace.append("GoalAgent")
        req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
        resp = GoalAgent().run(req)
        return resp, {"trace": trace, "route": decision.model_dump()}

    # default: finance education Q&A
    trace.append("FinanceQAAgent")
    req = _mk_req(user_text=user_text, user_profile=st.session_state["user_profile"], messages=msgs, extra=req_extra)
    resp = FinanceQAAgent().run(req)
    return resp, {"trace": trace, "route": decision.model_dump()}


# -----------------------------
# Sidebar (profile)
# -----------------------------


with st.sidebar:
    st.subheader("Profile")
    up: UserProfile = st.session_state["user_profile"]

    up.currency = st.text_input("Currency", value=up.currency)
    up.country = st.text_input("Country (optional)", value=up.country or "") or None
    up.risk_tolerance = st.selectbox("Risk tolerance", ["low", "medium", "high"], index=["low", "medium", "high"].index(up.risk_tolerance))
    up.knowledge_level = st.selectbox(
        "Knowledge level",
        ["beginner", "intermediate", "advanced"],
        index=["beginner", "intermediate", "advanced"].index(up.knowledge_level),
    )
    st.session_state["user_profile"] = up

    st.divider()
    st.caption(f"Session: {st.session_state['session_id']}")
    st.caption(f"Turn: {st.session_state['turn_id']}")


# -----------------------------
# Main UI
# -----------------------------


st.title("AI Finance Assistant")

tab_chat, tab_portfolio, tab_market, tab_goals = st.tabs(["Chat", "Portfolio", "Market", "Goals"])


# -----------------------------
# Chat tab
# -----------------------------


with tab_chat:
    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        st.subheader("Chat")

        # Render history
        for t in st.session_state["chat"]:
            turn = ChatTurn(**t)
            with st.chat_message(turn.role):
                st.markdown(turn.content)
                if turn.role == "assistant":
                    agent = turn.meta.get("agent_name")
                    if agent:
                        st.caption(f"Answered by: {agent}")
                    # Small inline meta
                    if turn.meta.get("warnings"):
                        st.caption("Warnings: " + ", ".join(turn.meta["warnings"]))

        # Input
        user_text = st.chat_input("Ask about markets, goals, portfolios, or concepts")
        if user_text:
            st.session_state["chat"].append({"role": "user", "content": user_text, "meta": {}})

            resp, meta = _run_chat_agent(user_text)

            st.session_state["last_chat_meta"] = {
                "agent_name": resp.agent_name,
                "citations": [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in (resp.citations or [])],
                "trace": meta.get("trace"),
                "route": meta.get("route"),
                "warnings": resp.warnings or [],
                "data_freshness": resp.data_freshness,
                "raw": resp.model_dump() if hasattr(resp, "model_dump") else {},
            }

            st.session_state["chat"].append(
                {
                    "role": "assistant",
                    "content": resp.answer_md,
                    "meta": {
                        "agent_name": resp.agent_name,
                        "citations": st.session_state["last_chat_meta"]["citations"],
                        "trace": meta.get("trace"),
                        "route": meta.get("route"),
                        "warnings": resp.warnings or [],
                        "data_freshness": resp.data_freshness,
                    },
                }
            )
            st.rerun()

    with right:
        st.subheader("Citations")
        meta = st.session_state.get("last_chat_meta") or {}
        _render_citations(meta.get("citations") or [])

        st.divider()
        st.subheader("Agent trace")
        _render_agent_trace(meta.get("trace"), meta.get("route"))


# -----------------------------
# Portfolio tab
# -----------------------------


def _parse_portfolio_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """Coerce a CSV into the portfolio payload expected by PortfolioAgent.

    Expected columns (case-insensitive):
    - symbol (required)
    - quantity (required; default 0)
    - asset_type (optional)
    - cash (optional; or a row with symbol=CASH)
    """
    if df is None or df.empty:
        return {"holdings": [], "cash": "0", "currency": st.session_state["user_profile"].currency}

    cols = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in cols:
        raise ValueError("CSV must have a 'symbol' column")

    sym_col = cols["symbol"]
    qty_col = cols.get("quantity")
    at_col = cols.get("asset_type")
    cash_col = cols.get("cash")

    cash = Decimal("0")
    holdings: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip()
        if not sym:
            continue

        if sym.upper() == "CASH":
            # quantity used as cash amount (optional)
            if qty_col and str(row.get(qty_col, "")).strip() != "":
                cash += Decimal(str(row.get(qty_col)))
            continue

        qty = Decimal("0")
        if qty_col and str(row.get(qty_col, "")).strip() != "":
            qty = Decimal(str(row.get(qty_col)))

        asset_type = "stock"
        if at_col and str(row.get(at_col, "")).strip() != "":
            asset_type = str(row.get(at_col)).strip().lower()

        holdings.append(
            {
                "symbol": sym,
                "quantity": str(qty),
                "asset_type": asset_type,
            }
        )

        if cash_col and str(row.get(cash_col, "")).strip() != "":
            cash += Decimal(str(row.get(cash_col)))

    return {"holdings": holdings, "cash": str(cash), "currency": st.session_state["user_profile"].currency}


with tab_portfolio:
    st.subheader("Portfolio dashboard")
    st.caption("Upload a CSV with columns: symbol, quantity, asset_type (optional), cash (optional).")

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], accept_multiple_files=False)

    col_a, col_b = st.columns([0.55, 0.45], gap="large")

    with col_a:
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df, use_container_width=True)

            if st.button("Analyze portfolio", type="primary"):
                try:
                    pf_payload = _parse_portfolio_csv(df)
                    req = AgentRequest(
                        request_id=str(uuid.uuid4()),
                        session_id=st.session_state["session_id"],
                        turn_id=int(st.session_state["turn_id"] + 1),
                        user_text="analyze portfolio",
                        user_profile=st.session_state["user_profile"],
                        market_payload={"portfolio": pf_payload},
                    )
                    resp = PortfolioAgent().run(req)
                    st.markdown(resp.answer_md)

                    metrics = (resp.data or {}).get("metrics") or {}
                    allocs = metrics.get("allocations") or []
                    st.session_state["_portfolio_allocs"] = allocs
                    st.session_state["_portfolio_metrics"] = metrics
                except Exception as e:
                    st.error(f"Portfolio analysis failed: {e}")
        else:
            st.info("Upload a CSV to compute metrics and charts.")

    with col_b:
        metrics = st.session_state.get("_portfolio_metrics") or {}
        allocs = st.session_state.get("_portfolio_allocs") or []
        if metrics:
            st.subheader("Metrics")
            st.metric("Total value", f"{metrics.get('total_value', 0):,.2f} {metrics.get('currency', '')}")
            st.metric("Risk bucket", str(metrics.get("risk_bucket", "-")))
            st.metric("Effective N", f"{metrics.get('diversification_effective_n', 0):.2f}")

        if allocs:
            st.subheader("Allocation")
            df_alloc = pd.DataFrame(allocs)
            df_alloc["weight_pct"] = (df_alloc["weight"].astype(float) * 100.0).round(2)

            fig_pie = px.pie(df_alloc, values="weight_pct", names="symbol", title="Allocation by symbol (%)")
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_bar = px.bar(
                df_alloc.sort_values("weight_pct", ascending=False).head(15),
                x="symbol",
                y="weight_pct",
                title="Top weights (%, up to 15)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        elif uploaded:
            st.caption("No allocation chart yet (run analysis).")


# -----------------------------
# Market tab
# -----------------------------


with tab_market:
    st.subheader("Market view")

    col1, col2 = st.columns([0.4, 0.6], gap="large")

    with col1:
        symbol = st.text_input("Ticker", value="AAPL")
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=0)
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
        force = st.checkbox("Force refresh", value=False)

        if st.button("Fetch", type="primary"):
            try:
                req = AgentRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=st.session_state["session_id"],
                    turn_id=int(st.session_state["turn_id"] + 1),
                    user_text=f"quote {symbol}",
                    user_profile=st.session_state["user_profile"],
                    market_payload={"symbol": symbol},
                )
                resp = MarketAgent().run(req)
                st.session_state["_market_resp"] = resp

                svc = MarketDataService()
                series = svc.get_history_close(symbol, period=period, interval=interval, force_refresh=force)
                st.session_state["_market_series"] = series
            except Exception as e:
                st.error(f"Market lookup failed: {e}")

    with col2:
        resp: Optional[AgentResponse] = st.session_state.get("_market_resp")
        if resp:
            st.markdown(resp.answer_md)

            q = resp.data or {}
            as_of = q.get("as_of")
            if isinstance(as_of, str):
                try:
                    as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                except Exception:
                    as_of_dt = None
            else:
                as_of_dt = as_of if isinstance(as_of, datetime) else None

            label, kind = _freshness_badge(
                as_of=as_of_dt,
                from_cache=bool(q.get("from_cache", False)),
                ttl_seconds=q.get("ttl_seconds"),
            )
            _badge(label, kind)

        series = st.session_state.get("_market_series")
        if series:
            s = series.model_dump() if hasattr(series, "model_dump") else dict(series)
            df = pd.DataFrame({"date": s.get("dates", []), "close": s.get("close", [])})
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                fig = px.line(df, x="date", y="close", title=f"{s.get('symbol', '')} close")
                st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Goals tab
# -----------------------------


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


with tab_goals:
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
