import chromadb
from sentence_transformers import SentenceTransformer
from researchpilot.config import CHROMA_PATH

class VectorStore:
    def __init__(self, db_path=None):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=str(db_path or CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(name="researchpilot")

    def replace_documents(self, documents: list[dict]):
        # Serialized by ResearchPilotService; local single-worker prototype.
        try:
            self.client.delete_collection("researchpilot")
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(name="researchpilot")
        if not documents:
            return
        for start in range(0, len(documents), 100):
            batch = documents[start:start + 100]
            embeddings = self.model.encode([d["text"] for d in batch],
                                           normalize_embeddings=True).tolist()
            self.collection.upsert(ids=[d["id"] for d in batch],
                                   documents=[d["text"] for d in batch],
                                   embeddings=embeddings,
                                   metadatas=[{"source": d["source"], "page": d["page"]}
                                              for d in batch])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        count = self.collection.count()
        if not count:
            return []
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=embedding,
                                       n_results=min(top_k, count),
                                       include=["documents", "metadatas", "distances"])
        return [{"id": key, "text": text, "source": meta["source"],
                 "page": meta["page"], "distance": dist}
                for key, text, meta, dist in zip(result["ids"][0],
                    result["documents"][0], result["metadatas"][0],
                    result["distances"][0])]
