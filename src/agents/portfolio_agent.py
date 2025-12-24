from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.utils.market_data import MarketDataService
from src.tools.quant_tools import tool_compute_portfolio_metrics


def _extract_portfolio(req: AgentRequest) -> Optional[Dict[str, Any]]:
    if getattr(req, "portfolio", None) is not None:
        p = req.portfolio
        return p.model_dump() if hasattr(p, "model_dump") else p.dict()

    mp = getattr(req, "market_payload", None)
    if isinstance(mp, dict) and isinstance(mp.get("portfolio"), dict):
        return mp["portfolio"]

    return None


class PortfolioAgent(BaseAgent):
    name = "portfolio_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            portfolio = _extract_portfolio(req)
            if not portfolio:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Please provide a portfolio in `AgentRequest.portfolio` or `market_payload.portfolio`.",
                    data={},
                    citations=[],
                    warnings=["MISSING_PORTFOLIO_PAYLOAD"],
                    confidence="low",
                )

            mkt = MarketDataService()
            prices: Dict[str, float] = {}
            for h in portfolio.get("holdings", []):
                sym = (h.get("symbol") or "").strip()
                if not sym or sym.upper() == "CASH":
                    continue
                try:
                    q = mkt.get_quote(sym)
                    if q and getattr(q, "last_price", None) is not None:
                        prices[sym] = float(q.last_price)
                except Exception:
                    continue

            metrics = tool_compute_portfolio_metrics(portfolio, prices=prices)

            top = (metrics.get("top_holdings") or [])[:5]
            top_lines = [f"- **{t['symbol']}**: {t['weight']*100:.1f}% (value {t['value']:.2f})" for t in top]

            answer_md = (
                "## Portfolio snapshot\n"
                f"- Total value: **{metrics['total_value']:.2f} {metrics['currency']}**\n"
                f"- Risk bucket (heuristic): **{metrics['risk_bucket']}**\n"
                "- Concentration (top 1 / top 3 / top 5): "
                f"**{metrics['concentration_top1']*100:.1f}% / {metrics['concentration_top3']*100:.1f}% / {metrics['concentration_top5']*100:.1f}%**\n"
                f"- Diversification (effective N): **{metrics['diversification_effective_n']:.2f}**\n\n"
                "### Top holdings\n"
                + ("\n".join(top_lines) if top_lines else "- (none)")
                + "\n\n### Notes\n"
                "- Numbers are computed deterministically (no LLM math).\n"
                "- If prices were missing for some symbols, add valid tickers or enable a provider.\n"
            )

            warnings = metrics.get("warnings") or []
            confidence = "high" if prices else "medium"

            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data={"prices_used": prices, "metrics": metrics},
                citations=[],
                warnings=warnings,
                confidence=confidence,
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="PortfolioAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
