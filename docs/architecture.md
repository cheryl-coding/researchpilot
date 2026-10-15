# ResearchPilot Architecture

```mermaid
flowchart TD
    UI[Streamlit] --> API[FastAPI]
    API --> ING[PDF ingestion]
    ING --> CH[Chunking + content hash IDs]
    CH --> BM[BM25 index]
    CH --> DB[(ChromaDB)]
    API --> AG[LangGraph agent]
    AG --> RET[BM25 / Dense / Hybrid]
    RET --> BM
    RET --> DB
    AG --> G[Evidence grading]
    G -->|Sufficient| A[LLM answer with citations]
    G -->|Insufficient and attempts remain| R[Query rewrite]
    R --> RET
    G -->|Retries exhausted| X[Abstain]
```

The service is a single-process prototype. Index rebuilds use an in-process lock, not a distributed transaction. It must not be deployed as an unauthenticated multi-user service. The evidence grader is probabilistic; valid citation IDs do not prove citation faithfulness.
