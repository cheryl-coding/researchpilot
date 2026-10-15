from researchpilot.agent.graph import build_agent, initial_state, route_after_grading

class FakeNodes:
    def __init__(self, succeed_at=None):
        self.succeed_at = succeed_at
    def retrieve(self, state):
        return {"attempts": state["attempts"] + 1, "retrieved_docs": []}
    def grade_evidence(self, state):
        return {"evidence_sufficient": state["attempts"] == self.succeed_at}
    def rewrite_query(self, state):
        return {"search_query": f"rewrite-{state['attempts']}"}
    def generate(self, state):
        return {"answer": "Answer [1]", "sources": [{"id": 1}]}
    def abstain(self, state):
        return {"answer": "Insufficient evidence", "sources": []}

def test_retries_then_answer():
    result = build_agent(FakeNodes(2)).invoke(initial_state("Question?"))
    assert result["attempts"] == 2
    assert result["answer"] == "Answer [1]"

def test_bounded_abstention():
    result = build_agent(FakeNodes()).invoke(initial_state("Question?", 3))
    assert result["attempts"] == 3
    assert result["sources"] == []

def test_routing():
    state = initial_state("Question?")
    state.update(evidence_sufficient=False, attempts=3)
    assert route_after_grading(state) == "abstain"
