import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = Path(os.getenv("RESEARCHPILOT_DATA_DIR", "data"))
PDF_DIR = DATA_DIR / "pdfs"
CORPUS_PATH = DATA_DIR / "corpus.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_results.json"
CHROMA_PATH = Path(os.getenv("RESEARCHPILOT_CHROMA_PATH", "chroma_db"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
