from __future__ import annotations

from decimal import Decimal, getcontext
from typing import Dict, List, Optional

from src.utils.quant_models import (
    PortfolioInput, PortfolioMetrics, AllocationRow,
    GoalInput, GoalProjection, ScenarioRow
)

getcontext().prec = 28

def _d(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))

def _safe_float(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.01")))

def _monthly_rate(r_annual: Decimal) -> Decimal:
    if r_annual < 0:
        raise ValueError("expected_return_annual cannot be negative")
    return (Decimal(1) + r_annual) ** (Decimal(1) / Decimal(12)) - Decimal(1)

def _future_value(current: Decimal, monthly: Decimal, n_months: int, mr: Decimal, stepup_annual_pct: Decimal) -> Decimal:
    if n_months <= 0:
        return current

    fv = current * ((Decimal(1) + mr) ** n_months)
    contrib = monthly

    for m in range(1, n_months + 1):
        if stepup_annual_pct > 0 and m % 12 == 1 and m != 1:
            contrib = contrib * (Decimal(1) + stepup_annual_pct)
        fv += contrib * ((Decimal(1) + mr) ** (n_months - m))
    return fv

def compute_portfolio_metrics(
    portfolio: PortfolioInput,
    prices: Optional[Dict[str, float]] = None,
    *,
    include_cash_row: bool = True,
) -> PortfolioMetrics:
    warnings: List[str] = []
    data_quality: Dict[str, str] = {}

    prices = prices or {}
    allocs: List[AllocationRow] = []

    total = _d(portfolio.cash)

    for h in portfolio.holdings:
        sym = (h.symbol or "").strip()
        if not sym:
            warnings.append("Encountered holding with empty symbol; skipped.")
            data_quality["empty_symbol"] = "present"
            continue

        px = prices.get(sym)
        if px is None:
            warnings.append(f"Missing price for {sym}; valued at 0. Provide prices from MarketDataService.")
            data_quality[f"missing_price:{sym}"] = "true"
            value = Decimal(0)
        else:
            value = _d(h.quantity) * _d(px) if px >= 0 else Decimal(0)

        total += value
        allocs.append(AllocationRow(symbol=sym, asset_type=h.asset_type, value=float(value), weight=0.0))

    if include_cash_row and portfolio.cash > 0:
        allocs.append(AllocationRow(symbol="CASH", asset_type="cash", value=float(_d(portfolio.cash)), weight=0.0))

    if total <= 0:
        warnings.append("Total portfolio value is 0. Check holdings/prices/cash.")
        return PortfolioMetrics(
            currency=portfolio.currency,
            total_value=0.0,
            allocations=[],
            top_holdings=[],
            concentration_top1=0.0,
            concentration_top3=0.0,
            concentration_top5=0.0,
            diversification_effective_n=0.0,
            risk_bucket="low",
            warnings=warnings,
            data_quality=data_quality,
        )

    total_dec = total
    for a in allocs:
        a.weight = float(Decimal(str(a.value)) / total_dec)

    sorted_allocs = sorted(allocs, key=lambda x: x.weight, reverse=True)
    top = [x for x in sorted_allocs if x.symbol != "CASH"][:10]

    def sum_top(n: int) -> float:
        return float(sum([x.weight for x in top[:n]])) if top else 0.0

    c1 = sum_top(1)
    c3 = sum_top(3)
    c5 = sum_top(5)

    hhi = sum([x.weight ** 2 for x in allocs]) if allocs else 0.0
    eff_n = (1.0 / hhi) if hhi > 0 else 0.0

    equity_like = sum([x.weight for x in allocs if x.asset_type in ("stock", "etf", "mutual_fund", "crypto")])

    if equity_like >= 0.75:
        risk = "high"
    elif equity_like >= 0.40:
        risk = "medium"
    else:
        risk = "low"

    return PortfolioMetrics(
        currency=portfolio.currency,
        total_value=_safe_float(total),
        allocations=sorted_allocs,
        top_holdings=top,
        concentration_top1=round(c1, 4),
        concentration_top3=round(c3, 4),
        concentration_top5=round(c5, 4),
        diversification_effective_n=round(eff_n, 4),
        risk_bucket=risk,  # type: ignore[arg-type]
        warnings=warnings,
        data_quality=data_quality,
    )

def compute_goal_projection(goal: GoalInput) -> GoalProjection:
    warnings: List[str] = []
    data_quality: Dict[str, str] = {}

    target = _d(goal.target_amount)
    years = _d(goal.years)
    n_months = int((years * Decimal(12)).to_integral_value(rounding="ROUND_HALF_UP"))

    r_annual = _d(goal.expected_return_annual)
    inf = _d(goal.inflation_annual)
    stepup = _d(goal.stepup_annual_pct)

    mr = _monthly_rate(r_annual)
    fv = _future_value(_d(goal.current_savings), _d(goal.monthly_contribution), n_months, mr, stepup)
    real = fv / ((Decimal(1) + inf) ** years) if inf > 0 else fv

    def reached(monthly: Decimal) -> Decimal:
        return _future_value(_d(goal.current_savings), monthly, n_months, mr, stepup)

    fv0 = reached(Decimal(0))
    if fv0 >= target:
        req = Decimal(0)
    else:
        lo = Decimal(0)
        hi = max(Decimal(10), target / Decimal(max(1, n_months)))
        for _ in range(40):
            if reached(hi) >= target:
                break
            hi *= Decimal(2)

        for _ in range(60):
            mid = (lo + hi) / Decimal(2)
            if reached(mid) >= target:
                hi = mid
            else:
                lo = mid
        req = hi

    def scenario(label: str, r: Decimal) -> ScenarioRow:
        mr_s = _monthly_rate(r)
        fv_s = _future_value(_d(goal.current_savings), _d(goal.monthly_contribution), n_months, mr_s, stepup)
        real_s = fv_s / ((Decimal(1) + inf) ** years) if inf > 0 else fv_s
        return ScenarioRow(
            label=label,
            expected_return_annual=float(r),
            projected_amount=_safe_float(fv_s),
            real_value_today=_safe_float(real_s),
        )

    scenarios = [
        scenario("low", max(Decimal(0), r_annual - Decimal("0.03"))),
        scenario("base", r_annual),
        scenario("high", r_annual + Decimal("0.03")),
    ]

    return GoalProjection(
        currency=goal.currency,
        target_amount=_safe_float(target),
        years=float(years),
        assumptions={
            "expected_return_annual": float(r_annual),
            "inflation_annual": float(inf),
            "stepup_annual_pct": float(stepup),
        },
        projected_amount=_safe_float(fv),
        real_value_today=_safe_float(real),
        required_monthly_for_target=_safe_float(req),
        scenarios=scenarios,
        warnings=warnings,
        data_quality=data_quality,
    )
