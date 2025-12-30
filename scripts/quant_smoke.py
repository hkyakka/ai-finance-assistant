from __future__ import annotations

from src.tools.quant_tools import tool_compute_portfolio_metrics, tool_compute_goal_projection

def main():
    portfolio = {
        "currency": "USD",
        "cash": "500",
        "holdings": [
            {"symbol": "AAPL", "quantity": "3", "asset_type": "stock"},
            {"symbol": "VOO", "quantity": "5", "asset_type": "etf"},
            {"symbol": "BND", "quantity": "10", "asset_type": "bond"},
        ],
    }
    prices = {"AAPL": 200.0, "VOO": 450.0, "BND": 70.0}
    pm = tool_compute_portfolio_metrics(portfolio, prices=prices)
    print("Portfolio total:", pm["total_value"], pm["currency"])
    print("Top holding concentration:", pm["concentration_top1"])
    print("Effective N:", pm["diversification_effective_n"])
    print("Risk bucket:", pm["risk_bucket"])

    goal = {
        "currency": "USD",
        "target_amount": "100000",
        "years": "10",
        "current_savings": "5000",
        "monthly_contribution": "500",
        "expected_return_annual": "0.10",
        "inflation_annual": "0.06",
        "stepup_annual_pct": "0.05",
    }
    gp = tool_compute_goal_projection(goal)
    print("Projected amount:", gp["projected_amount"])
    print("Real value today:", gp["real_value_today"])
    print("Required monthly for target:", gp["required_monthly_for_target"])
    for s in gp["scenarios"]:
        print("Scenario:", s["label"], s["projected_amount"])

if __name__ == "__main__":
    main()
