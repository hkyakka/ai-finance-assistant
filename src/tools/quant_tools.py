from __future__ import annotations

from typing import Dict, Any, Optional
from src.utils.quant_engine import compute_portfolio_metrics, compute_goal_projection
from src.utils.quant_models import PortfolioInput, GoalInput

def tool_compute_portfolio_metrics(payload: Dict[str, Any], prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    p = PortfolioInput(**payload)
    out = compute_portfolio_metrics(p, prices=prices)
    return out.model_dump()

def tool_compute_goal_projection(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload or {})

    # map common aliases -> canonical fields expected by GoalInput (quant schema)
    if "years" not in p and "time_horizon_years" in p:
        p["years"] = p["time_horizon_years"]

    if "initial_investment" not in p and "current_savings" in p:
        p["initial_investment"] = p["current_savings"]

    if "monthly_investment" not in p and "monthly_contribution" in p:
        p["monthly_investment"] = p["monthly_contribution"]

    if "expected_return" not in p and "expected_return_annual" in p:
        p["expected_return"] = p["expected_return_annual"]

    if "inflation_pct" not in p and "inflation_annual" in p:
        p["inflation_pct"] = p["inflation_annual"]

    if "risk_tolerance" not in p and "risk_profile" in p:
        p["risk_tolerance"] = p["risk_profile"]

    g = GoalInput(**p)
    out = compute_goal_projection(g)
    return out.model_dump()
