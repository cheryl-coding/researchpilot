import math
import re
from collections import Counter, defaultdict


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.documents, self.k1, self.b = documents, k1, b
        self.N = len(documents)
        self.term_freqs = [Counter(tokenize(d["text"])) for d in documents]
        self.doc_lengths = [sum(c.values()) for c in self.term_freqs]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 0
        self.doc_freq = defaultdict(int)
        for counts in self.term_freqs:
            for term in counts:
                self.doc_freq[term] += 1

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str, index: int) -> float:
        if not self.avgdl:
            return 0.0
        counts = self.term_freqs[index]
        length = self.doc_lengths[index]
        score = 0.0
        for term in set(tokenize(query)):
            f = counts.get(term, 0)
            if f:
                score += (
                    self.idf(term)
                    * (f * (self.k1 + 1))
                    / (f + self.k1 * (1 - self.b + self.b * length / self.avgdl))
                )
        return score

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [dict(doc, score=self.score(query, i)) for i, doc in enumerate(self.documents)]
        return sorted((d for d in scored if d["score"] > 0), key=lambda d: (-d["score"], d["id"]))[
            :top_k
        ]
