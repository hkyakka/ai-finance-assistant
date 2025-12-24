from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from typing import Any, Dict, List, Optional


def _d(x: Any, default: str = "0") -> Decimal:
    if x is None:
        return Decimal(default)
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal(default)


def _round_money(x: Decimal, nd: int = 2) -> Decimal:
    q = Decimal("1").scaleb(-nd)  # 10^-nd
    return x.quantize(q, rounding=ROUND_HALF_UP)


def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if not isfinite(f):
            return None
        return f
    except Exception:
        return None


def _risk_return_assumption(risk: str) -> float:
    # Conservative baseline; tweak later as you calibrate
    r = (risk or "medium").lower().strip()
    if r == "low":
        return 0.06
    if r == "high":
        return 0.12
    return 0.10


def tool_compute_goal_projection(goal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic goal projection tool.
    Accepts both new and legacy keys:
      - time_horizon_years OR years
      - monthly_investment OR monthly_contribution
      - initial_investment OR current_savings
      - inflation_pct OR inflation_annual
    """
    warnings: List[str] = []

    if not isinstance(goal, dict):
        return {"warnings": ["BAD_GOAL_INPUT_TYPE"], "error": "goal must be a dict"}

    currency = (goal.get("currency") or "USD").upper()
    goal_name = goal.get("goal_name") or "My Goal"

    target_amount = _d(goal.get("target_amount") or goal.get("target") or "0")
    years = goal.get("time_horizon_years", None)
    if years is None:
        years = goal.get("years", None)

    try:
        years = int(years)
    except Exception:
        years = None

    if years is None or years <= 0:
        return {"warnings": ["MISSING_YEARS"], "error": "time_horizon_years/years is required and must be > 0"}

    monthly = goal.get("monthly_investment", None)
    if monthly is None:
        monthly = goal.get("monthly_contribution", None)
    monthly = _d(monthly or "0")

    initial = goal.get("initial_investment", None)
    if initial is None:
        initial = goal.get("current_savings", None)
    initial = _d(initial or "0")

    inflation = goal.get("inflation_pct", None)
    if inflation is None:
        inflation = goal.get("inflation_annual", None)
    inflation_pct = _safe_float(inflation) if inflation is not None else None
    if inflation_pct is None:
        inflation_pct = 0.05  # default 5%
        warnings.append("DEFAULT_INFLATION_5PCT")

    risk = (goal.get("risk_tolerance") or "medium").lower()
    annual_return = _risk_return_assumption(risk)

    n_months = years * 12
    r_month = annual_return / 12.0

    # FV of SIP + initial
    if r_month == 0:
        fv_sip = monthly * Decimal(n_months)
        fv_initial = initial
    else:
        # (1+r)^n
        growth = Decimal(str((1 + r_month) ** n_months))
        fv_initial = initial * growth
        # monthly * (( (1+r)^n - 1 ) / r )
        fv_sip = monthly * (growth - Decimal("1")) / Decimal(str(r_month))

    projected = fv_initial + fv_sip
    projected = _round_money(projected)

    # Real value in today's money (discount by inflation)
    real = projected / Decimal(str((1 + inflation_pct) ** years))
    real = _round_money(real)

    # Required SIP to reach target (given assumptions), ignoring step-ups
    req_monthly: Optional[Decimal] = None
    if target_amount > 0:
        if r_month == 0:
            req_monthly = (target_amount - initial) / Decimal(n_months)
        else:
            growth = Decimal(str((1 + r_month) ** n_months))
            denom = (growth - Decimal("1")) / Decimal(str(r_month))
            req_monthly = (target_amount - (initial * growth)) / denom if denom != 0 else None
        if req_monthly is not None:
            if req_monthly < 0:
                req_monthly = Decimal("0")
            req_monthly = _round_money(req_monthly)
    else:
        warnings.append("TARGET_AMOUNT_ZERO_OR_MISSING")

    # Scenarios: +/- 3% return
    scenarios = []
    for delta in (-0.03, 0.0, 0.03):
        rr = max(0.0, annual_return + delta)
        rm = rr / 12.0
        if rm == 0:
            scen = initial + monthly * Decimal(n_months)
        else:
            growth = Decimal(str((1 + rm) ** n_months))
            scen = initial * growth + monthly * (growth - Decimal("1")) / Decimal(str(rm))
        scenarios.append({"annual_return": rr, "projected_amount": float(_round_money(scen))})

    return {
        "goal_name": goal_name,
        "currency": currency,
        "target_amount": float(target_amount),
        "years": years,
        "monthly_investment": float(monthly),
        "initial_investment": float(initial),
        "assumed_annual_return": annual_return,
        "assumed_inflation_pct": inflation_pct,
        "projected_amount": float(projected),
        "projected_real_value_today": float(real),
        "required_monthly_for_target": float(req_monthly) if req_monthly is not None else None,
        "scenarios": scenarios,
        "as_of": datetime.utcnow().isoformat(),
        "warnings": warnings,
    }


def tool_compute_portfolio_metrics(portfolio: Dict[str, Any], prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Deterministic portfolio metrics: totals, weights, concentration, diversification.
    portfolio schema (dict-like):
      - holdings: [{symbol, quantity, asset_type, ...}, ...]
      - cash: number
      - currency: string
    prices: {symbol: last_price}
    """
    warnings: List[str] = []

    if not isinstance(portfolio, dict):
        return {"warnings": ["BAD_PORTFOLIO_INPUT_TYPE"], "error": "portfolio must be a dict"}

    currency = (portfolio.get("currency") or "USD").upper()
    holdings = portfolio.get("holdings") or []
    cash = _d(portfolio.get("cash") or "0")

    prices = prices or {}

    # value each holding
    valued = []
    total = cash
    for h in holdings:
        sym = (h.get("symbol") or "").upper().strip()
        if not sym:
            continue
        qty = _d(h.get("quantity") or "0")
        asset_type = (h.get("asset_type") or "stock").lower()

        if sym == "CASH" or asset_type == "cash":
            total += qty
            continue

        px = prices.get(sym) or prices.get(sym.upper())
        if px is None:
            warnings.append(f"Missing price for {sym}; valued at 0. Provide prices from MarketDataService.")
            val = Decimal("0")
        else:
            val = _d(px) * qty

        total += val
        valued.append({"symbol": sym, "value": val, "asset_type": asset_type})

    total_f = float(_round_money(total))
    if total <= 0:
        return {
            "currency": currency,
            "total_value": 0.0,
            "risk_bucket": "low",
            "top_holdings": [],
            "concentration_top1": 0.0,
            "concentration_top3": 0.0,
            "concentration_top5": 0.0,
            "diversification_effective_n": 0.0,
            "warnings": warnings + ["TOTAL_VALUE_ZERO"],
        }

    # weights and effective N
    for v in valued:
        v["weight"] = float(v["value"] / total) if total != 0 else 0.0

    valued_sorted = sorted(valued, key=lambda x: x["weight"], reverse=True)

    wts = [v["weight"] for v in valued_sorted if v["weight"] > 0]
    hh = sum([w * w for w in wts]) if wts else 1.0
    effective_n = 1.0 / hh if hh > 0 else 1.0

    top1 = sum(wts[:1]) if len(wts) >= 1 else 0.0
    top3 = sum(wts[:3]) if len(wts) >= 3 else sum(wts)
    top5 = sum(wts[:5]) if len(wts) >= 5 else sum(wts)

    # heuristic risk: equity-like weight
    equity_like = 0.0
    for v in valued_sorted:
        at = v["asset_type"]
        if at in ("stock", "etf", "crypto", "mf"):
            equity_like += v["weight"]
    if equity_like >= 0.75:
        risk_bucket = "high"
    elif equity_like >= 0.40:
        risk_bucket = "medium"
    else:
        risk_bucket = "low"

    top_holdings = [
        {"symbol": v["symbol"], "weight": float(v["weight"]), "value": float(_round_money(v["value"]))}
        for v in valued_sorted[:10]
    ]

    return {
        "currency": currency,
        "total_value": total_f,
        "risk_bucket": risk_bucket,
        "concentration_top1": float(top1),
        "concentration_top3": float(top3),
        "concentration_top5": float(top5),
        "diversification_effective_n": float(effective_n),
        "top_holdings": top_holdings,
        "warnings": warnings,
        "as_of": datetime.utcnow().isoformat(),
    }
