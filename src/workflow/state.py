from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from src.core.schemas import (
    AgentResponse,
    AgentTraceEvent,
    ErrorEnvelope,
    GoalInput,
    PortfolioInput,
    QuantResult,
    RagResult,
    RouterDecision,
    ToolCall,
    UserProfile,
)


class GraphState(TypedDict, total=False):
    """LangGraph state shape (dict-based)."""
    request_id: str
    session_id: str
    turn_id: int

    user_text: str
    user_profile: UserProfile

    # conversational memory
    memory_summary: str
    memory_turns: List[Dict[str, str]]  # [{user, assistant}]

    route: RouterDecision

    agent_trace: List[str]
    trace_events: List[AgentTraceEvent]
    tool_calls: List[ToolCall]

    portfolio: PortfolioInput
    goal: GoalInput

    rag_result: RagResult
    quant_result: QuantResult
    market_payload: Dict[str, Any]

    responses: List[AgentResponse]
    final: AgentResponse

    error: ErrorEnvelope


class ConversationStateModel(BaseModel):
    request_id: str = ""
    session_id: str = ""
    turn_id: int = 0

    user_text: str = ""
    user_profile: UserProfile = Field(default_factory=UserProfile)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Memory
    memory_summary: str = ""
    memory_turns: List[Dict[str, str]] = Field(default_factory=list)

    route: Optional[RouterDecision] = None

    agent_trace: List[str] = Field(default_factory=list)
    trace_events: List[AgentTraceEvent] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)

    portfolio: Optional[PortfolioInput] = None
    goal: Optional[GoalInput] = None

    rag_result: Optional[RagResult] = None
    quant_result: Optional[QuantResult] = None
    market_payload: Dict[str, Any] = Field(default_factory=dict)

    responses: List[AgentResponse] = Field(default_factory=list)
    final: Optional[AgentResponse] = None

    error: Optional[ErrorEnvelope] = None

    def to_graph_state(self) -> GraphState:
        return self.model_dump()  # type: ignore[return-value]

    @classmethod
    def from_graph_state(cls, state: GraphState) -> "ConversationStateModel":
        return cls(**state)
