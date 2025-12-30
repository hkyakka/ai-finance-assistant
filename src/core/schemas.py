from __future__ import annotations

from datetime import datetime, UTC
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


# -------------------------
# Common / Core
# -------------------------

class Citation(BaseModel):
    doc_id: str
    title: str
    url: str
    snippet: str
    score: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UserProfile(BaseModel):
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    knowledge_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    country: Optional[str] = None
    currency: str = "USD"


# -------------------------
# Portfolio
# -------------------------

class Holding(BaseModel):
    symbol: str
    quantity: Decimal = Field(default=Decimal("0"))
    avg_cost: Optional[Decimal] = None
    asset_type: Literal["stock", "etf", "bond", "cash", "crypto", "mf"] = "stock"
    expense_ratio: Optional[float] = None


class PortfolioInput(BaseModel):
    holdings: List[Holding] = Field(default_factory=list)
    cash: Decimal = Field(default=Decimal("0"))
    currency: str = "USD"


# -------------------------
# Goals
# -------------------------

class GoalInput(BaseModel):
    goal_name: str = "My Goal"
    target_amount: Decimal
    time_horizon_years: int = Field(ge=1, le=60)
    monthly_investment: Optional[Decimal] = None
    initial_investment: Optional[Decimal] = None
    inflation_pct: Optional[float] = None
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    currency: str = "USD"


# -------------------------
# Market Data
# -------------------------

class MarketQuote(BaseModel):
    symbol: str
    price: float
    currency: str = "USD"
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: Optional[str] = None
    from_cache: bool = False
    ttl_seconds: Optional[int] = None

    # Backward/forward compatibility for agents that use `last_price`
    @property
    def last_price(self) -> float:
        return float(self.price)


class PriceSeries(BaseModel):
    symbol: str
    dates: List[str]           # ISO date strings
    close: List[float]
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: Optional[str] = None
    from_cache: bool = False


# -------------------------
# RAG
# -------------------------

class RagChunk(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    url: str
    snippet: str
    score: float


class RagResult(BaseModel):
    query: str
    chunks: List[RagChunk] = Field(default_factory=list)


# -------------------------
# Quant (Deterministic Computation)
# -------------------------

class QuantTaskType(str, Enum):
    PORTFOLIO_METRICS = "PORTFOLIO_METRICS"
    GOAL_PROJECTION = "GOAL_PROJECTION"
    MARKET_INDICATORS = "MARKET_INDICATORS"
    REBALANCE_SIM = "REBALANCE_SIM"


class DataQuality(BaseModel):
    missing_pct: float = 0.0
    dropped_dates: int = 0
    stale_data: bool = False
    notes: List[str] = Field(default_factory=list)


class QuantRequest(BaseModel):
    task_type: QuantTaskType
    portfolio: Optional[PortfolioInput] = None
    goal: Optional[GoalInput] = None
    symbols: List[str] = Field(default_factory=list)
    date_range: Optional[Tuple[str, str]] = None  # ("YYYY-MM-DD", "YYYY-MM-DD")
    frequency: Literal["1d", "1wk", "1mo"] = "1d"
    currency: str = "USD"
    risk_tolerance: Literal["low", "medium", "high"] = "medium"

    # Policies
    rounding_currency_decimals: int = 2
    rounding_pct_decimals: int = 2
    max_ffill_gap_days: int = 3


class QuantResult(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tables: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    chart_data: Dict[str, Any] = Field(default_factory=dict)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    warnings: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


# -------------------------
# Agent responses
# -------------------------


# -------------------------
# Tooling + Errors + Tracing (Stage 2)
# -------------------------

class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retriable: bool = False


ToolName = Literal[
    "RAG_RETRIEVE",
    "MARKET_QUOTE",
    "MARKET_SERIES",
    "NEWS_SEARCH",
    "QUANT_COMPUTE",
    "GLOSSARY_LOOKUP",
]


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    args: dict = {}
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: Literal["started", "ok", "error"] = "started"
    result: Optional["ToolResult"] = None
    error: Optional[ErrorEnvelope] = None



class ToolResult(BaseModel):
    call_id: str
    tool_name: ToolName
    result: Optional[Any] = None
    error: Optional[ErrorEnvelope] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentTraceEvent(BaseModel):
    node: str
    agent: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    info: Dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    request_id: str
    session_id: str
    turn_id: int

    user_text: str
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # optional structured inputs
    portfolio: Optional[PortfolioInput] = None
    goal: Optional[GoalInput] = None

    # upstream system outputs
    route: Optional[RouterDecision] = None
    rag_result: Optional[RagResult] = None
    quant_result: Optional[QuantResult] = None
    market_payload: Optional[Dict[str, Any]] = None

    # memory / chat history
    messages: List[ChatMessage] = Field(default_factory=list)
    memory_summary: Optional[str] = None

class AgentResponse(BaseModel):
    """Standard agent output.

    Several stages (and the smoke scripts) attach `data` + `error`.
    Make the model permissive to avoid breaking when agents evolve.
    """

    model_config = ConfigDict(extra="allow")

    agent_name: str
    answer_md: str
    data: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Citation] = Field(default_factory=list)
    charts_payload: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    data_freshness: Optional[Dict[str, Any]] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    error: Optional[Dict[str, Any]] = None


# -------------------------
# Routing
# -------------------------

Intent = Literal["FINANCE_QA", "PORTFOLIO", "MARKET", "GOAL", "NEWS", "TAX", "CLARIFY"]


class RouterDecision(BaseModel):
    intent: Intent
    confidence: float = 0.7
    secondary_intents: List[Intent] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    rationale: Optional[str] = None
