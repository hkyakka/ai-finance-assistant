import uuid
from src.core.schemas import AgentRequest, UserProfile, GoalInput
from src.agents.goal_agent import GoalAgent
from src.agents.portfolio_agent import PortfolioAgent

def test_goal_agent_contract():
    up = UserProfile()
    goal = GoalInput(
        target_amount=10000,
        time_horizon_years=5,
        monthly_investment=100,
        risk_tolerance="medium"
    )
    out = GoalAgent().run(AgentRequest(user_text="goal", user_profile=up, goal=goal, request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()), turn_id=1))
    assert out.agent_name == "goal_agent"
    assert "Goal projection" in out.answer_md
    assert out.data.get("projection")

def test_portfolio_agent_missing_payload():
    up = UserProfile()
    out = PortfolioAgent().run(AgentRequest(user_text="portfolio", user_profile=up, payload={}, request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()), turn_id=1))
    assert out.agent_name == "portfolio_agent"
    assert out.confidence == "low"
