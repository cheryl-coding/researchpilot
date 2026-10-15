from typing import Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    max_attempts: int = Field(default=3, ge=1, le=5)

class Citation(BaseModel):
    id: int
    document: str
    page: int
    text: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    attempts: int
    trace: list[str]

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str

class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
