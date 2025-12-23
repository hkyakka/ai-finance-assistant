from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse


class NewsAgent(BaseAgent):
    name = "NewsAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        return AgentResponse(
            agent_name=self.name,
            answer_md='(TODO) News summary (Stage 5/7).',
            citations=[],
            warnings=["Agent stub (Stage 2). Implement in later stages."],
            confidence="low",
        )
