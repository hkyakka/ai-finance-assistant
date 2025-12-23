from src.utils.quant_engine import compute_portfolio_metrics
from src.utils.quant_models import PortfolioInput

def test_portfolio_metrics_basic():
    p = PortfolioInput(
        currency="USD",
        cash="100",
        holdings=[
            {"symbol": "AAA", "quantity": "2", "asset_type": "stock"},
            {"symbol": "BBB", "quantity": "1", "asset_type": "bond"},
        ],
    )
    prices = {"AAA": 50.0, "BBB": 100.0}
    out = compute_portfolio_metrics(p, prices=prices)
    assert out.total_value == 300.00
    assert out.concentration_top1 > 0
    assert out.diversification_effective_n > 0

def test_missing_price_warning():
    p = PortfolioInput(currency="USD", cash="0", holdings=[{"symbol": "AAA", "quantity": "1", "asset_type": "stock"}])
    out = compute_portfolio_metrics(p, prices={})
    assert out.total_value == 0.00
    assert any("Missing price" in w for w in out.warnings)
