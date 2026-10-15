import threading
from pathlib import Path
from researchpilot.config import PDF_DIR, CHROMA_PATH, CORPUS_PATH, MODEL
from researchpilot.vector_store import VectorStore
from researchpilot.retrieval.corpus import build_corpus, save_corpus
from researchpilot.retrieval.bm25 import BM25Retriever
from researchpilot.retrieval.hybrid import HybridRetriever
from researchpilot.agent.nodes import AgentNodes
from researchpilot.agent.graph import build_agent, initial_state

class ResearchPilotService:
    """Single-process prototype: serialized reads and index rebuilds."""
    def __init__(self, pdf_dir=PDF_DIR, db_path=CHROMA_PATH):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.vector_store = VectorStore(db_path)
        self.corpus = []
        self.bm25 = BM25Retriever([])
        self.rebuild_indexes()

    def rebuild_indexes(self):
        with self.lock:
            docs = build_corpus(self.pdf_dir)
            # Not atomic; do not use for concurrent/multi-worker public deployment.
            self.vector_store.replace_documents(docs)
            self.corpus = docs
            self.bm25 = BM25Retriever(docs)
            save_corpus(docs, CORPUS_PATH)
            return len(docs)

    def get_retriever(self, mode):
        if mode == "bm25":
            return self.bm25
        if mode == "dense":
            return self.vector_store
        if mode == "hybrid":
            return HybridRetriever(self.bm25, self.vector_store)
        raise ValueError("Unknown retrieval mode")

    def ask(self, question, retrieval_mode="hybrid", max_attempts=3):
        with self.lock:
            nodes = AgentNodes(self.get_retriever(retrieval_mode), model=MODEL)
            return build_agent(nodes).invoke(initial_state(question, max_attempts))

    def list_documents(self):
        return [{"filename": p.name, "size_bytes": p.stat().st_size}
                for p in sorted(self.pdf_dir.glob("*.pdf"))]
