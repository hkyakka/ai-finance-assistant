from __future__ import annotations

import re
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.utils.market_data import MarketDataService

_TICKER_RE = re.compile(r"\b[A-Z]{1,6}(?:\.[A-Z]{1,3})?\b")


def _extract_symbol(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    m = _TICKER_RE.search(t)
    return m.group(0) if m else None


class MarketAgent(BaseAgent):
    name = "market_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            symbol = None
            mp = getattr(req, "market_payload", None)
            if isinstance(mp, dict):
                symbol = mp.get("symbol")
            if not symbol:
                symbol = _extract_symbol(req.user_text or "")

            if not symbol:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Tell me a ticker (e.g., **AAPL**, **MSFT**, **RELIANCE.NS**).",
                    data={},
                    citations=[],
                    warnings=["NO_SYMBOL"],
                    confidence="low",
                )

            mkt = MarketDataService()
            q = mkt.get_quote(symbol)

            answer_md = (
                f"## Market snapshot: **{symbol}**\n"
                f"- Last price: **{q.last_price}**\n"
                f"- Currency: **{q.currency or '-'}**\n"
                f"- Provider: **{q.provider}**\n"
                f"- As of: **{q.as_of}**\n"
                f"- From cache: **{q.from_cache}**\n"
            )

            data = q.model_dump() if hasattr(q, "model_dump") else q.__dict__
            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data=data,
                citations=[],
                warnings=[],
                confidence="high",
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="MarketAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
