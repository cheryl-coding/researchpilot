from typing import TypedDict
class AgentState(TypedDict):
    question: str
    search_query: str
    retrieved_docs: list[dict]
    evidence_sufficient: bool
    grading_reason: str
    attempts: int
    max_attempts: int
    answer: str
    sources: list[dict]
    trace: list[str]
