from langgraph.graph import StateGraph, START, END
from researchpilot.agent.state import AgentState

def route_after_grading(state: AgentState) -> str:
    if state["evidence_sufficient"]:
        return "generate"
    return "abstain" if state["attempts"] >= state["max_attempts"] else "rewrite_query"

def build_agent(nodes):
    graph = StateGraph(AgentState)
    for name in ("retrieve", "grade_evidence", "rewrite_query", "generate", "abstain"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_evidence")
    graph.add_conditional_edges("grade_evidence", route_after_grading,
                                {"generate": "generate", "abstain": "abstain",
                                 "rewrite_query": "rewrite_query"})
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()

def initial_state(question: str, max_attempts: int = 3) -> AgentState:
    if not question.strip() or not 1 <= max_attempts <= 5:
        raise ValueError("Invalid question or retry count")
    return {"question": question, "search_query": question, "retrieved_docs": [],
            "evidence_sufficient": False, "grading_reason": "", "attempts": 0,
            "max_attempts": max_attempts, "answer": "", "sources": [], "trace": []}
