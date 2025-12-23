from src.utils.quant_engine import compute_goal_projection
from src.utils.quant_models import GoalInput

def test_goal_projection_basic():
    g = GoalInput(
        currency="USD",
        target_amount="10000",
        years="5",
        current_savings="0",
        monthly_contribution="100",
        expected_return_annual="0.10",
        inflation_annual="0.05",
        stepup_annual_pct="0.00",
    )
    out = compute_goal_projection(g)
    assert out.projected_amount > 0
    assert out.required_monthly_for_target >= 0
    assert len(out.scenarios) == 3
