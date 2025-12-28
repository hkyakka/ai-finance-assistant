from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
from datetime import datetime
from typing import Optional

from src.agents.market_agent import MarketAgent
from src.core.schemas import AgentRequest, AgentResponse
from src.utils.market_data import MarketDataService
from src.web_app.ui_helpers import _freshness_badge, _badge

def render():
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
