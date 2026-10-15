from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60,
                           top_k: int = 5) -> list[dict]:
    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")
    scores = defaultdict(float)
    documents = {}
    for ranked in ranked_lists:
        seen = set()
        for rank, doc in enumerate(ranked, 1):
            key = doc["id"]
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, doc)
            scores[key] += 1 / (k + rank)
    ids = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [dict(documents[key], rrf_score=scores[key]) for key in ids]

class HybridRetriever:
    def __init__(self, bm25, dense, rrf_k: int = 60, candidate_k: int = 20):
        self.bm25, self.dense = bm25, dense
        self.rrf_k, self.candidate_k = rrf_k, candidate_k

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        n = max(top_k, self.candidate_k)
        return reciprocal_rank_fusion([self.bm25.search(query, n),
                                       self.dense.search(query, n)], self.rrf_k, top_k)
