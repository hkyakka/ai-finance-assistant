from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse


class TaxAgent(BaseAgent):
    name = "TaxAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        return AgentResponse(
            agent_name=self.name,
            answer_md='(TODO) Tax education answer (use RAG).',
            citations=[],
            warnings=["Agent stub (Stage 2). Implement in later stages."],
            confidence="low",
        )
