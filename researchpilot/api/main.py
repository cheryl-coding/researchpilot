import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from researchpilot.api.schemas import ChatRequest, ChatResponse, DocumentInfo, UploadResponse
from researchpilot.api.services import ResearchPilotService
from researchpilot.config import BENCHMARK_PATH

MAX_PDF_BYTES = 20 * 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ResearchPilotService()
    yield

app = FastAPI(title="ResearchPilot API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return app.state.service.list_documents()

def save_and_index(filename: str, content: bytes):
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted or not document.page_count:
                raise ValueError("Encrypted or empty PDF")
            if not any(page.get_text("text").strip() for page in document):
                raise ValueError("PDF contains no extractable text")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc
    service = app.state.service
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem)[:60] or "document"
    saved_name = f"{stem}_{uuid.uuid4().hex[:12]}.pdf"
    destination = service.pdf_dir / saved_name
    with service.lock:
        try:
            destination.write_bytes(content)
            count = service.rebuild_indexes()
        except Exception:
            destination.unlink(missing_ok=True)
            try:
                service.rebuild_indexes()
            except Exception:
                logging.exception("Could not restore index after failed upload")
            raise
    return UploadResponse(filename=saved_name, chunks_indexed=count, status="indexed")

@app.post("/documents/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported")
    content = await file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 20 MB")
    return await run_in_threadpool(save_and_index, file.filename, content)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await run_in_threadpool(app.state.service.ask, request.question,
                                         request.retrieval_mode, request.max_attempts)
        return ChatResponse(question=request.question, answer=result["answer"],
                            sources=result["sources"], attempts=result["attempts"],
                            trace=result["trace"])
    except Exception as exc:
        logging.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Unable to process question") from exc

@app.get("/evaluation/results")
def evaluation_results():
    if not BENCHMARK_PATH.exists():
        return {"results": []}
    return {"results": json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))}
