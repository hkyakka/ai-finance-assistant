from __future__ import annotations

import re
from typing import Optional

from src.core.schemas import RouterDecision, Intent


class Router:
    """
    Deterministic routing (good for consistency + tests).
    Later you can replace/augment classify() with an LLM call (temperature=0).
    """

    def __init__(self) -> None:
        # Simple keyword rules (tune as you learn evaluator prompts)
        self._rules = [
            ("TAX", r"\b(tax|income tax|income|earnings|withholding|withhold|ltcg|stcg|capital gain|capital gains|dividend tax|tds)\b"),
            ("NEWS", r"\b(news|headline|breaking|why is.*down|why is.*up)\b"),
            ("GOAL", r"\b(goal|retire|retirement|target|years|months|sip|required sip|plan)\b"),
            ("PORTFOLIO", r"\b(portfolio|holdings|allocation|diversif|rebalance|weights|overexposed)\b"),
            ("MARKET", r"\b(price|quote|trend|chart|volatility|moving average|rsi|market)\b"),
            ("FINANCE_QA", r"\b(what is|explain|difference between|define|how does)\b"),
        ]

    def classify(self, user_text: str) -> RouterDecision:
        text = (user_text or "").strip().lower()

        if not text:
            return RouterDecision(
                intent="CLARIFY",
                confidence=0.2,
                clarifying_question="What would you like to do: learn a concept, analyze a portfolio, check a stock, or plan a goal?"
            )

        # Quick ambiguity check: if user says "this/that" without context
        if re.search(r"\b(this|that|it|same as earlier)\b", text) and len(text.split()) <= 6:
            return RouterDecision(
                intent="CLARIFY",
                confidence=0.3,
                clarifying_question="Can you share which symbol/portfolio/goal you mean (or paste the details again)?",
                rationale="Pronoun-only reference without enough context."
            )

        matched: list[Intent] = []
        for intent, pattern in self._rules:
            if re.search(pattern, text):
                matched.append(intent)  # type: ignore[arg-type]

        if not matched:
            return RouterDecision(
                intent="FINANCE_QA",
                confidence=0.55,
                rationale="Defaulting to Finance Q&A due to no strong keyword match."
            )

        # Primary + possible secondaries (e.g., MARKET + NEWS)
        primary = matched[0]
        secondaries = list(dict.fromkeys(matched[1:]))[:2]

        return RouterDecision(
            intent=primary,
            confidence=0.75 if len(matched) == 1 else 0.7,
            secondary_intents=secondaries,
            rationale=f"Matched intents via keyword rules: {matched}"
        )