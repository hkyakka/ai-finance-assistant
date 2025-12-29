import streamlit as st
import pandas as pd
import plotly.express as px
from decimal import Decimal
from typing import Any, Dict, List
import uuid

from src.web_app.agent_helpers import _run_chat_agent
from src.core.schemas import AgentRequest

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


def render():
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
                    resp, _ = _run_chat_agent("analyze portfolio", "Portfolio")
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
