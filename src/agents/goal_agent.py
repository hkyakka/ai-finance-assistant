from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.tools.quant_tools import tool_compute_goal_projection


def _extract_goal(req: AgentRequest) -> Optional[Dict[str, Any]]:
    # Prefer structured field if present
    if getattr(req, "goal", None) is not None:
        g = req.goal
        return g.model_dump() if hasattr(g, "model_dump") else g.dict()

    # Fallback: market_payload dict
    mp = getattr(req, "market_payload", None)
    if isinstance(mp, dict) and isinstance(mp.get("goal"), dict):
        return mp["goal"]

    return None


class GoalAgent(BaseAgent):
    name = "goal_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            goal = _extract_goal(req)
            if not goal:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Please provide a goal in `AgentRequest.goal` (preferred) or `market_payload.goal`.",
                    data={},
                    citations=[],
                    warnings=["MISSING_GOAL_PAYLOAD"],
                    confidence="low",
                )

            proj = tool_compute_goal_projection(goal)

            # Robust scenario formatting: tolerate different keys
            scenarios = proj.get("scenarios") or []
            scen_lines = []
            for s in scenarios:
                label = s.get("label") or s.get("scenario") or s.get("name")
                rr = s.get("expected_return_annual")
                if rr is None:
                    rr = s.get("annual_return")
                if label is None:
                    if rr is None:
                        label = "Scenario"
                    else:
                        label = f"{float(rr) * 100:.1f}% return"
                if rr is None:
                    rr = 0.0

                amt = s.get("projected_amount")
                if amt is None:
                    amt = s.get("end_value") or s.get("future_value") or 0.0

                scen_lines.append(f"- **{label}** ({float(rr) * 100:.1f}%): {float(amt):.2f}")

            # Robustly pick real value key (tool returns projected_real_value_today)
            real_today = proj.get("real_value_today")
            if real_today is None:
                real_today = proj.get("projected_real_value_today")
            if real_today is None:
                real_today = 0.0

            years = proj.get("years")
            if years is None:
                years = proj.get("time_horizon_years")
            if years is None:
                years = 0.0

            answer_md = (
                "## Goal projection\n"
                f"- Target: **{float(proj.get('target_amount', 0.0)):.2f} {proj.get('currency', '')}** in **{float(years):.1f} years**\n"
                f"- Projected amount (base): **{float(proj.get('projected_amount', 0.0)):.2f} {proj.get('currency', '')}**\n"
                f"- Value in today's money: **{float(real_today):.2f} {proj.get('currency', '')}**\n"
                f"- Required monthly to reach target: **{float(proj.get('required_monthly_for_target', 0.0)):.2f} {proj.get('currency', '')}**\n\n"
                "### Scenarios\n"
                + ("\n".join(scen_lines) if scen_lines else "- (none)")
                + "\n\n### Notes\n"
                "- Computation is deterministic (no LLM math).\n"
                "- Scenarios are +/- return bands around base return.\n"
            )

            warnings = proj.get("warnings") or []
            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data={"projection": proj},
                citations=[],
                warnings=warnings,
                confidence="high",
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="GoalAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
