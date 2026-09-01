from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    strategy: str = Field(default="rerank")  # simple, semantic, sparse, hybrid, rerank, hyde, query_expansion, mmr, hybrid_rerank
    top_k: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = None  # groq, openai, anthropic, ollama
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    mmr_lambda: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    session_id: Optional[str] = None


class SourceCitation(BaseModel):
    source_file: str
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    embed_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    gen_ms: Optional[float] = None
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    strategy: str
    num_context_chunks: int
    retrieval_scores: list[float]
    grounded: bool = True
    grounding_note: Optional[str] = None
    processing_time_ms: float
    latencies: Optional[LatencyBreakdown] = None
    provider_used: Optional[str] = None


class FeedbackRequest(BaseModel):
    query_id: Optional[int] = None
    query_text: str
    rating: int = Field(..., ge=-1, le=1)  # 1 = positive, -1 = negative
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class DocumentItem(BaseModel):
    id: int
    title: str
    source_file: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total_docs: int
    total_chunks: int


class IngestRequest(BaseModel):
    strategy: str = 'simple'
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    clear_existing: bool = False
    incremental: bool = True


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_created: int
    strategy: str
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    postgres: str
    groq_api: str
    corpus_files: int
    active_strategies_count: int = 8
    local_only_mode: bool = False
    timestamp: datetime


class EvalResultsResponse(BaseModel):
    runs: list[dict]
    latest_comparison: Optional[list] = None

