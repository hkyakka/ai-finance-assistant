from __future__ import annotations

import uuid
from typing import Any

from src.core.schemas import AgentRequest, UserProfile, PortfolioInput, Holding, GoalInput, AgentResponse
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.tax_agent import TaxAgent
from src.agents.market_agent import MarketAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.goal_agent import GoalAgent
from src.agents.news_agent import NewsAgent


def mk_req(user_text: str, up: UserProfile, turn_id: int, *, portfolio=None, goal=None, market_payload=None) -> AgentRequest:
    return AgentRequest(
        request_id=str(uuid.uuid4()),
        session_id="smoke-session",
        turn_id=turn_id,
        user_text=user_text,
        user_profile=up,
        portfolio=portfolio,
        goal=goal,
        market_payload=market_payload or {},
    )


def _print_resp(title: str, resp: AgentResponse) -> None:
    print(f"--- {title} ---")
    print(resp.answer_md[:800])
    if resp.warnings:
        print("warnings:", resp.warnings)
    err = getattr(resp, "error", None)
    if err:
        print("error:", err)


def main():
    up = UserProfile()
    t = 1

    print("=== FinanceQA ===")
    _print_resp("FinanceQA", FinanceQAAgent().run(mk_req("What is an ETF?", up, t))); t += 1

    print("\n=== Tax ===")
    _print_resp("Tax", TaxAgent().run(mk_req("What is LTCG vs STCG?", up, t))); t += 1

    print("\n=== Market ===")
    _print_resp("Market", MarketAgent().run(mk_req("AAPL quote", up, t, market_payload={"symbol": "AAPL"}))); t += 1

    print("\n=== Portfolio ===")
    portfolio = PortfolioInput(
        currency="USD",
        cash="100",
        holdings=[
            Holding(symbol="AAPL", quantity="2", asset_type="stock"),
            Holding(symbol="BND", quantity="5", asset_type="bond"),
        ],
    )
    _print_resp("Portfolio", PortfolioAgent().run(mk_req("Analyze my portfolio", up, t, portfolio=portfolio))); t += 1

    print("\n=== Goal ===")
    # NOTE: GoalInput schema uses time_horizon_years + monthly_investment + initial_investment.
    goal = GoalInput(
        goal_name="100k in 10 years",
        currency="USD",
        target_amount="100000",
        time_horizon_years=10,
        initial_investment="5000",
        monthly_investment="500",
        inflation_pct=0.06,
        risk_tolerance="medium",
    )
    _print_resp("Goal", GoalAgent().run(mk_req("Can I reach 100k in 10 years?", up, t, goal=goal))); t += 1

    print("\n=== News ===")
    _print_resp("News", NewsAgent().run(mk_req("AAPL earnings", up, t, market_payload={"query": "AAPL earnings"}))); t += 1


if __name__ == "__main__":
    main()
