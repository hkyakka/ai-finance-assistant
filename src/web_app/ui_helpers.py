from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import streamlit as st
import pandas as pd

from typing import Any, Dict, List
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



def _render_agent_trace(trace: List[str] | None, router_decision: Any = None) -> None:
    """Render the LangGraph trace and router decision.

    router_decision may be:
      - dict (preferred)
      - str (route name like 'FINANCE_QA')
      - None
    """
    rd: Dict[str, Any] = {}
    if isinstance(router_decision, dict):
        rd = router_decision
    elif isinstance(router_decision, str):
        rd = {"intent": router_decision, "confidence": 1.0}
    elif router_decision is not None:
        # Best-effort coercion (e.g., pydantic model)
        if hasattr(router_decision, "model_dump"):
            try:
                rd = router_decision.model_dump()  # type: ignore[attr-defined]
            except Exception:
                rd = {}
        else:
            rd = {}

    if rd:
        intent = rd.get("intent") or rd.get("route") or rd.get("agent") or "unknown"
        conf = rd.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        st.markdown(f"**Route:** `{intent}` (conf {conf_f:.2f})")
        rationale = rd.get("rationale") or rd.get("reason")
        if rationale:
            st.caption(str(rationale))

    if not trace:
        st.caption("No trace")
        return
    st.code("\n".join([f"- {t}" for t in trace]), language="text")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

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
