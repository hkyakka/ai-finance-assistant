from src.workflow.graph import build_graph
from src.core.schemas import UserProfile

def main():
    graph = build_graph()
    state = {"user_text": "Analyze my portfolio risk", "user_profile": UserProfile()}
    out = graph.invoke(state)

    print("\n=== FINAL ANSWER ===")
    print(out["final"].answer_md)

    print("\n=== TRACE ===")
    print(out.get("agent_trace"))

if __name__ == "__main__":
    main()
