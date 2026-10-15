from dataclasses import dataclass

from researchpilot.pdf_loader import Page


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_number: int


def chunk_pages(pages: list[Page], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")
    chunks = []
    for page in pages:
        words = page.text.split()
        for start in range(0, len(words), chunk_size - overlap):
            selected = words[start : start + chunk_size]
            if selected:
                chunks.append(
                    Chunk(
                        f"{page.source}:{page.page_number}:{start}",
                        " ".join(selected),
                        page.source,
                        page.page_number,
                    )
                )
            if start + chunk_size >= len(words):
                break
    return chunks
