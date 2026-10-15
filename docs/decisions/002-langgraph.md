# ADR-002: Bounded LangGraph agent

**Status:** Accepted

We use an explicit state graph for retrieve → grade → (generate | rewrite | abstain). This is testable and bounds retrieval attempts. Grading can make mistakes; retrying increases API cost. Deterministic routing tests use fake nodes and no network calls.
