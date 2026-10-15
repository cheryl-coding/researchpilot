def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0

def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    for i, key in enumerate(retrieved_ids[:k], 1):
        if key in relevant_ids:
            return 1 / i
    return 0.0
