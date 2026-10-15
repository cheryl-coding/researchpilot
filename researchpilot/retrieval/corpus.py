import hashlib
import json
from pathlib import Path

from researchpilot.chunker import chunk_pages
from researchpilot.config import CORPUS_PATH, PDF_DIR
from researchpilot.pdf_loader import load_pdf


def build_corpus(pdf_directory: str | Path = PDF_DIR) -> list[dict]:
    documents = []
    for path in sorted(Path(pdf_directory).glob("*.pdf")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for chunk in chunk_pages(load_pdf(path)):
            # The final suffix is the start-word offset from the original chunk ID.
            offset = chunk.chunk_id.rsplit(":", 1)[-1]
            documents.append(
                {
                    "id": f"{digest}:{chunk.page_number}:{offset}",
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page_number,
                }
            )
    return documents


def save_corpus(documents: list[dict], path: str | Path = CORPUS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, indent=2), encoding="utf-8")


def load_corpus(path: str | Path = CORPUS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
