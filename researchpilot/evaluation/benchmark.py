import json
import statistics
import time

from researchpilot.config import BENCHMARK_PATH, CHROMA_PATH, CORPUS_PATH, DATA_DIR
from researchpilot.evaluation.metrics import recall_at_k, reciprocal_rank_at_k
from researchpilot.retrieval.bm25 import BM25Retriever
from researchpilot.retrieval.corpus import load_corpus
from researchpilot.retrieval.hybrid import HybridRetriever
from researchpilot.vector_store import VectorStore


def evaluate(retriever, questions, k=5):
    recalls, mrrs, latencies = [], [], []
    for item in questions:
        start = time.perf_counter()
        results = retriever.search(item["question"], top_k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [doc["id"] for doc in results]
        relevant = set(item["relevant_ids"])
        recalls.append(recall_at_k(ids, relevant, k))
        mrrs.append(reciprocal_rank_at_k(ids, relevant, k))
    return {
        "recall_at_5": statistics.mean(recalls),
        "mrr_at_5": statistics.mean(mrrs),
        "latency_ms": statistics.mean(latencies),
    }


def main():
    corpus = load_corpus(CORPUS_PATH)
    questions = json.loads((DATA_DIR / "benchmark_questions.json").read_text())
    if not questions:
        raise ValueError("Add real, labeled benchmark questions first")
    ids = {d["id"] for d in corpus}
    for q in questions:
        if not q["relevant_ids"] or not set(q["relevant_ids"]).issubset(ids):
            raise ValueError(f"Invalid relevance labels: {q['id']}")
    bm25 = BM25Retriever(corpus)
    dense = VectorStore(CHROMA_PATH)
    if set(dense.collection.get(include=[])["ids"]) != ids:
        raise RuntimeError("Chroma and corpus IDs differ; reindex before benchmarking")
    rows = []
    for name, retriever in {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid": HybridRetriever(bm25, dense),
    }.items():
        rows.append({"method": name, **evaluate(retriever, questions)})
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
