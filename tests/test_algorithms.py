import pytest
from researchpilot.pdf_loader import Page
from researchpilot.chunker import chunk_pages
from researchpilot.retrieval.bm25 import BM25Retriever, tokenize
from researchpilot.retrieval.hybrid import reciprocal_rank_fusion
from researchpilot.evaluation.metrics import recall_at_k, reciprocal_rank_at_k

def docs():
    return [{"id": "a", "text": "transformer attention mechanism", "source": "a.pdf", "page": 1},
            {"id": "b", "text": "convolutional filters", "source": "b.pdf", "page": 2}]

def test_chunk_overlap():
    chunks = chunk_pages([Page("one two three four five six seven eight", "x.pdf", 1)], 5, 2)
    assert [c.text for c in chunks] == ["one two three four five", "four five six seven eight"]

def test_bm25():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]
    assert BM25Retriever(docs()).search("attention")[0]["id"] == "a"
    assert BM25Retriever(docs()).search("nonexistent") == []

def test_rrf():
    a, b = docs()
    result = reciprocal_rank_fusion([[a, b], [b]])
    assert result[0]["id"] == "b"
    assert len({r["id"] for r in result}) == len(result)

def test_metrics():
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == pytest.approx(.5)
    assert reciprocal_rank_at_k(["a", "b"], {"b"}, 2) == .5
