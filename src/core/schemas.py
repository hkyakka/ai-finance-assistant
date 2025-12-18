from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


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
    ts: datetime = Field(default_factory=datetime.utcnow)


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
    as_of: datetime = Field(default_factory=datetime.utcnow)
    provider: Optional[str] = None
    from_cache: bool = False
    ttl_seconds: Optional[int] = None


class PriceSeries(BaseModel):
    symbol: str
    dates: List[str]           # ISO date strings
    close: List[float]
    as_of: datetime = Field(default_factory=datetime.utcnow)
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

class AgentResponse(BaseModel):
    agent_name: str
    answer_md: str
    citations: List[Citation] = Field(default_factory=list)
    charts_payload: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    data_freshness: Optional[Dict[str, Any]] = None
    confidence: Literal["high", "medium", "low"] = "medium"


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
