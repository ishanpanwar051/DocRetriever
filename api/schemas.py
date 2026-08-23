from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    strategy: Literal['simple', 'semantic', 'hybrid', 'rerank'] = 'simple'
    top_k: int = Field(default=5, ge=1, le=20)

class SourceCitation(BaseModel):
    source_file: str
    section_title: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    strategy: str
    num_context_chunks: int
    retrieval_scores: list[float]
    processing_time_ms: float

class IngestRequest(BaseModel):
    strategy: Literal['simple', 'semantic', 'hybrid', 'rerank'] = 'simple'
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    clear_existing: bool = False

class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_created: int
    strategy: str
    processing_time_seconds: float

class HealthResponse(BaseModel):
    status: str
    postgres: str
    ollama: str
    corpus_files: int
    timestamp: datetime

class EvalResultsResponse(BaseModel):
    runs: list[dict]
    latest_comparison: Optional[dict] = None
