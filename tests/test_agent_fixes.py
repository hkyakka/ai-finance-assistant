from src.agents.tax_agent import TaxAgent
from src.agents.news_agent import NewsAgent
from src.core.schemas import AgentRequest
from dotenv import load_dotenv
import os

def run_tests():
    load_dotenv()
    test_tax_agent()
    test_news_agent()

def test_tax_agent():
    agent = TaxAgent()
    req = AgentRequest(
        user_text="What is Capital Gains?",
        request_id="test-req-1",
        session_id="test-session-1",
        turn_id=1,
    )
    resp = agent.run(req)

    print("\n--- Tax Agent Test ---")
    print(f"Query: {req.user_text}")
    print(f"Response:\n{resp.answer_md}")

    assert "capital gain" in resp.answer_md.lower(), "Expected 'capital gain' in the response"
    assert resp.warnings == [], f"Expected no warnings, but got {resp.warnings}"


def test_news_agent():
    agent = NewsAgent()
    req = AgentRequest(
        user_text="Apple Earnings",
        request_id="test-req-2",
        session_id="test-session-2",
        turn_id=2,
    )
    resp = agent.run(req)

    print("\n--- News Agent Test ---")
    print(f"Query: {req.user_text}")
    print(f"Response:\n{resp.answer_md}")

    assert "summary" in resp.data, "Expected summary in News Agent data"
    assert "Summary of news for" in resp.answer_md, "Expected summary in News Agent response"
    assert resp.warnings == [], f"Expected no warnings, but got {resp.warnings}"

if __name__ == "__main__":
    run_tests()
