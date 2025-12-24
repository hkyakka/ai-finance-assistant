from src.core.schemas import AgentRequest, UserProfile
from src.agents.goal_agent import GoalAgent
from src.agents.portfolio_agent import PortfolioAgent

def test_goal_agent_contract():
    up = UserProfile()
    payload = {"goal": {"currency":"USD","target_amount":"10000","years":"5","current_savings":"0","monthly_contribution":"100","expected_return_annual":"0.10","inflation_annual":"0.05","stepup_annual_pct":"0"}}
    out = GoalAgent().run(AgentRequest(user_text="goal", user_profile=up, payload=payload))
    assert out.agent_name == "goal_agent"
    assert "Goal projection" in out.answer_md
    assert out.data.get("projection")

def test_portfolio_agent_missing_payload():
    up = UserProfile()
    out = PortfolioAgent().run(AgentRequest(user_text="portfolio", user_profile=up, payload={}))
    assert out.agent_name == "portfolio_agent"
    assert out.confidence == 0.0
