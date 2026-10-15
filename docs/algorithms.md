# Algorithms

## BM25
The lexical retriever implements a smoothed-IDF BM25 ranking function using term frequencies, corpus document frequencies and document-length normalization. Query-time scoring scans the corpus; an inverted index is a potential optimization.

## Dense retrieval
Sentence Transformers `all-MiniLM-L6-v2` embeds passages and queries, and ChromaDB retrieves nearest neighbors. BM25 and dense search use the same chunk IDs.

## Reciprocal Rank Fusion
Hybrid retrieval merges ranked results using `sum(1 / (k + rank))` for `k=60`. Scores are based on ranks rather than adding incompatible BM25 and embedding scores.

## Agentic routing
LangGraph retrieves, grades, rewrites queries when necessary, and stops after at most three retrieval attempts by default.
