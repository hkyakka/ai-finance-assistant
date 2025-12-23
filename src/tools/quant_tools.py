from __future__ import annotations

from typing import Dict, Any, Optional
from src.utils.quant_engine import compute_portfolio_metrics, compute_goal_projection
from src.utils.quant_models import PortfolioInput, GoalInput

def tool_compute_portfolio_metrics(payload: Dict[str, Any], prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    p = PortfolioInput(**payload)
    out = compute_portfolio_metrics(p, prices=prices)
    return out.model_dump()

def tool_compute_goal_projection(payload: Dict[str, Any]) -> Dict[str, Any]:
    g = GoalInput(**payload)
    out = compute_goal_projection(g)
    return out.model_dump()
