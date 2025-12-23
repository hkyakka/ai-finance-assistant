from __future__ import annotations

from abc import ABC, abstractmethod
from src.core.schemas import AgentRequest, AgentResponse

class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def run(self, req: AgentRequest) -> AgentResponse:
        raise NotImplementedError
