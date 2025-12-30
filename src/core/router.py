from __future__ import annotations

import json
import os
import re
from typing import Optional, List, Dict, Any

from src.core.schemas import RouterDecision, Intent
from src.core.llm_client import LLMClient


_TAB_ALIASES: Dict[str, Intent] = {
    "news": "NEWS",
    "tax": "TAX",
    "market": "MARKET",
    "goals": "GOAL",
    "goal": "GOAL",
    "portfolio": "PORTFOLIO",
}

# Legacy / fallback keyword rules (kept as a safety net)
_RULES: List[tuple[Intent, re.Pattern[str]]] = [
    ("TAX", re.compile(r"\b(tax|capital\s*gain|stcg|ltcg|irs|hmrc|gst|tds|withholding)\b", re.I)),
    ("MARKET", re.compile(r"\b(price|quote|market\s*snapshot|cmp|last\s*price)\b", re.I)),
    ("PORTFOLIO", re.compile(r"\b(portfolio|holdings?|allocation|rebalance|diversif)\b", re.I)),
    ("GOAL", re.compile(r"\b(goal|retire|retirement|target\s*amount|time\s*horizon)\b", re.I)),
    ("NEWS", re.compile(r"\b(news|headline|article|breaking|what\s*happened)\b", re.I)),
]


class Router:
    """LLM-assisted router with deterministic overrides.

    - If query comes from a deterministic tab (News/Market/Goals/Tax/Portfolio), routing is forced.
    - For the Chat/Finance-QA tab, we ask the LLM to pick the best intent among allowed candidates.
    - If LLM is unavailable (missing keys/deps), we fall back to keyword rules.
    """

    def __init__(self) -> None:
        # Keep a tiny, precompiled fallback route map
        self._rules = _RULES

    def classify(self, user_text: str) -> RouterDecision:
        # Back-compat for older callers that didn't pass source_tab.
        return self.decide(user_text=user_text, source_tab=None)

    def decide(
        self,
        *,
        user_text: str,
        source_tab: Optional[str] = None,
        has_goal: bool = False,
        has_portfolio: bool = False,
    ) -> RouterDecision:
        # Structured objects win (graph passes these flags)
        if has_goal:
            return RouterDecision(intent="GOAL", confidence=0.99, rationale="Explicit goal input present.")
        if has_portfolio:
            return RouterDecision(intent="PORTFOLIO", confidence=0.99, rationale="Explicit portfolio input present.")

        tab_key = (source_tab or "").strip().lower()
        if tab_key in _TAB_ALIASES:
            forced = _TAB_ALIASES[tab_key]
            return RouterDecision(intent=forced, confidence=0.99, rationale=f"Forced by tab={source_tab!r}.")

        # For chat-like tab, attempt LLM routing with safe fallback.
        return self._llm_or_fallback(user_text=user_text, tab=tab_key or "chat")

    def _llm_or_fallback(self, *, user_text: str, tab: str) -> RouterDecision:
        # Candidates depend on tab. For now, chat can route to finance/tax/market/portfolio.
        allowed: List[Intent] = ["FINANCE_QA", "TAX", "MARKET", "PORTFOLIO", "GOAL"]
        # Never route to NEWS from chat (news has its own tab/agent)
        allowed = [i for i in allowed if i != "NEWS"]

        # Quick rule-based short-circuit for obvious tax/market requests (still reduces cost)
        rule_hit = self._rules_route(user_text)
        if rule_hit and rule_hit.intent in allowed:
            return rule_hit

        # Explicit opt-out for environments without keys.
        if (os.getenv("ROUTER_MODE") or "").strip().lower() in ("rules", "deterministic", "off"):
            return self._rules_route(user_text) or RouterDecision(intent="FINANCE_QA", confidence=0.6)

        try:
            llm = LLMClient(temperature=0.0)
            prompt = self._router_prompt(user_text=user_text, tab=tab, allowed=allowed)
            out = llm.generate(prompt).text
            return self._parse_router_json(out, allowed=allowed)
        except Exception as e:
            # Safe fallback
            fallback = self._rules_route(user_text) or RouterDecision(intent="FINANCE_QA", confidence=0.55)
            fallback.rationale = (fallback.rationale or "") + f" (LLM router unavailable: {e})"
            return fallback

    def _rules_route(self, user_text: str) -> Optional[RouterDecision]:
        text = (user_text or "").strip()
        if not text:
            return RouterDecision(intent="CLARIFY", confidence=0.4, clarify_question="What would you like to know?")
        matched: List[Intent] = []
        for intent, pat in self._rules:
            if pat.search(text):
                matched.append(intent)
        if not matched:
            return None
        primary = matched[0]
        secondaries = list(dict.fromkeys(matched[1:]))[:2]
        return RouterDecision(intent=primary, confidence=0.75, secondary_intents=secondaries, rationale=f"Rules matched: {matched}")

    def _router_prompt(self, *, user_text: str, tab: str, allowed: List[Intent]) -> str:
        # Keep JSON-only output for easy parsing.
        allowed_str = ", ".join(allowed)
        return (
            "You are an intent router for a finance assistant.\n"
            f"The query comes from tab='{tab}'.\n"
            f"Choose exactly one intent from: {allowed_str}.\n\n"
            "Intent guidance:\n"
            "- TAX: taxes, capital gains, withholding, filings, country-specific tax questions.\n"
            "- MARKET: price/quote requests or very short market snapshot questions.\n"
            "- PORTFOLIO: holdings/allocation/rebalancing, portfolio risk, diversification of a specific portfolio.\n"
            "- GOAL: retirement/goal planning, target amount, time horizon, SIP/contribution planning.\n"
            "- FINANCE_QA: finance concepts, investing education, general explanations.\n\n"
            "Return ONLY valid JSON with keys: intent, confidence, rationale.\n"
            "confidence must be a number in [0,1].\n\n"
            f"Query: {user_text}\n"
        )

    def _parse_router_json(self, text: str, *, allowed: List[Intent]) -> RouterDecision:
        raw = (text or "").strip()
        # Try to locate a JSON object in the model output.
        if not raw.startswith("{"):
            m = re.search(r"\{.*\}", raw, flags=re.S)
            raw = m.group(0) if m else raw
        try:
            obj: Dict[str, Any] = json.loads(raw)
        except Exception:
            # If parsing fails, fall back.
            return self._rules_route(text) or RouterDecision(intent="FINANCE_QA", confidence=0.55, rationale="Router JSON parse failed; defaulting.")

        intent = str(obj.get("intent") or "").strip().upper()
        if intent not in allowed:
            intent = "FINANCE_QA"
        try:
            conf = float(obj.get("confidence") or 0.6)
        except Exception:
            conf = 0.6
        conf = min(1.0, max(0.0, conf))
        rationale = str(obj.get("rationale") or "").strip() or None
        return RouterDecision(intent=intent, confidence=conf, rationale=rationale)
