from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from researchpilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
)
from researchpilot.api.services import ResearchPilotService

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ResearchPilotService()
    yield


app = FastAPI(
    title="ResearchPilot API",
    description="Agentic RAG research assistant for academic PDFs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents")
async def list_documents():
    service = app.state.service
    return service.list_documents()


@app.post("/documents/upload", response_model=UploadResponse)
async def upload(file: Annotated[UploadFile, File()]):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF",
        )

    service = app.state.service

    saved_name = f"{uuid4().hex}_{Path(file.filename).name}"
    destination = service.pdf_dir / saved_name

    try:
        destination.write_bytes(content)

        count = await run_in_threadpool(service.rebuild_indexes)

    except Exception:
        logger.exception("Failed to index uploaded PDF")

        destination.unlink(missing_ok=True)

        try:
            await run_in_threadpool(service.rebuild_indexes)
        except Exception:
            logger.exception("Could not restore index after failed upload")

        raise HTTPException(
            status_code=500,
            detail="Unable to process uploaded PDF",
        ) from None

    return UploadResponse(
        filename=saved_name,
        chunks_indexed=count,
        status="indexed",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await run_in_threadpool(
            app.state.service.ask,
            request.question,
            request.retrieval_mode,
            request.max_attempts,
        )

        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            attempts=result["attempts"],
            trace=result["trace"],
        )

    except Exception as exc:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process question",
        ) from exc


@app.get("/evaluation/results")
async def evaluation_results():
    service = app.state.service
    return service.evaluation_results()
