from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse


class GoalAgent(BaseAgent):
    name = "GoalAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        return AgentResponse(
            agent_name=self.name,
            answer_md='(TODO) Goal planning answer (use Quant Agent projections in Stage 6).',
            citations=[],
            warnings=["Agent stub (Stage 2). Implement in later stages."],
            confidence="low",
        )
