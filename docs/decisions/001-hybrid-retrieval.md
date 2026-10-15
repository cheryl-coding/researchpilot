# ADR-001: Hybrid retrieval with RRF

**Status:** Accepted

BM25 and dense retrieval are complementary. We combine their ranks using RRF because raw lexical and embedding scores are not directly comparable. Trade-offs include two indexes, higher latency and dependence on candidate count. Validate against the same labeled corpus using Recall@5, MRR@5 and latency.
