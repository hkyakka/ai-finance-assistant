from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.tools.quant_tools import tool_compute_portfolio_metrics, tool_compute_goal_projection

class QuantAgent(BaseAgent):
    name = "quant_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            payload = req.payload or {}
            kind = payload.get("kind")
            if kind == "portfolio":
                portfolio = payload.get("portfolio") or {}
                prices = payload.get("prices") or None
                result = tool_compute_portfolio_metrics(portfolio, prices=prices)
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="(Quant) Portfolio metrics computed.",
                    data=result,
                    citations=[],
                    warnings=[],
                    confidence=1.0,
                )
            if kind == "goal":
                goal = payload.get("goal") or {}
                result = tool_compute_goal_projection(goal)
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="(Quant) Goal projection computed.",
                    data=result,
                    citations=[],
                    warnings=[],
                    confidence=1.0,
                )

            return AgentResponse(
                agent_name=self.name,
                answer_md="QuantAgent expects payload.kind in {'portfolio','goal'}.",
                data={},
                citations=[],
                warnings=["BAD_INPUT_KIND"],
                confidence=0.0,
                error=ErrorEnvelope(code="BAD_INPUT", message="Missing/invalid payload.kind").model_dump(),
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="QuantAgent failed to compute.",
                data={},
                citations=[],
                warnings=["COMPUTE_FAILED"],
                confidence=0.0,
                error=ErrorEnvelope(code="COMPUTE_FAILED", message=str(e)).model_dump(),
            )
