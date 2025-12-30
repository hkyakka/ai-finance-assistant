from src.workflow.graph import build_graph
from src.core.schemas import UserProfile, PortfolioInput, Holding

def main():
    graph = build_graph()
    state = {
        "request_id": "smoke",
        "session_id": "smoke",
        "turn_id": 1,
        "user_text": "Analyze my portfolio risk",
        "user_profile": UserProfile(),
        "portfolio": PortfolioInput(currency="USD", holdings=[Holding(symbol="AAPL", quantity=2, avg_cost=90)]),
    }
    out = graph.invoke(state)

    print("\n=== FINAL ANSWER ===")
    print(out["final"].answer_md)

    print("\n=== TRACE ===")
    print(out.get("agent_trace"))

    print("\n=== TOOL CALLS ===")
    for tc in out.get("tool_calls", []):
        print(f"- {tc.tool_name}: {tc.status}")

if __name__ == "__main__":
    main()
