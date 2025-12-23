from __future__ import annotations

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, condecimal

AssetType = Literal["stock", "etf", "mutual_fund", "bond", "cash", "crypto", "other"]

class Holding(BaseModel):
    symbol: str = Field(..., description="Ticker/symbol. Use 'CASH' for cash rows if needed.")
    quantity: condecimal(ge=0) = Field(..., description="Units/shares. Decimal supported.")
    asset_type: AssetType = "other"
    avg_cost: Optional[condecimal(ge=0)] = None
    currency: Optional[str] = None

class PortfolioInput(BaseModel):
    currency: str = "USD"
    holdings: List[Holding] = Field(default_factory=list)
    cash: condecimal(ge=0) = 0

class AllocationRow(BaseModel):
    symbol: str
    asset_type: AssetType
    value: float
    weight: float

class PortfolioMetrics(BaseModel):
    currency: str
    total_value: float
    allocations: List[AllocationRow]
    top_holdings: List[AllocationRow]
    concentration_top1: float
    concentration_top3: float
    concentration_top5: float
    diversification_effective_n: float = Field(..., description="1/sum(w^2) (effective number of holdings)")
    risk_bucket: Literal["low", "medium", "high"]
    warnings: List[str] = Field(default_factory=list)
    data_quality: Dict[str, str] = Field(default_factory=dict)

class GoalInput(BaseModel):
    target_amount: condecimal(gt=0)
    years: condecimal(gt=0)
    current_savings: condecimal(ge=0) = 0
    monthly_contribution: condecimal(ge=0) = 0
    expected_return_annual: condecimal(ge=0) = 0.10
    inflation_annual: condecimal(ge=0) = 0.06
    stepup_annual_pct: condecimal(ge=0) = 0
    currency: str = "USD"

class ScenarioRow(BaseModel):
    label: str
    expected_return_annual: float
    projected_amount: float
    real_value_today: float

class GoalProjection(BaseModel):
    currency: str
    target_amount: float
    years: float
    assumptions: Dict[str, float]
    projected_amount: float
    real_value_today: float
    required_monthly_for_target: float
    scenarios: List[ScenarioRow]
    warnings: List[str] = Field(default_factory=list)
    data_quality: Dict[str, str] = Field(default_factory=dict)
