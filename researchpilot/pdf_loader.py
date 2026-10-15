from dataclasses import dataclass
from pathlib import Path
import fitz

@dataclass
class Page:
    text: str
    source: str
    page_number: int

def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    with fitz.open(path) as pdf:
        return [Page(text, path.name, i + 1) for i, p in enumerate(pdf)
                if (text := p.get_text("text").strip())]
