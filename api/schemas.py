"""
api/schemas.py — Pydantic Request & Response Schemas for DocuMind API Platform
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class CitationItem(BaseModel):
    page_number: int = Field(default=1, description="1-indexed document page number")
    source: str = Field(..., description="Document filename or identifier")
    text_excerpt: str = Field(..., description="Exact retrieved passage or table snippet")
    relevance_score: float = Field(default=0.0, description="Cosine similarity or cross-encoder score")
    is_table: bool = Field(default=False, description="Whether the passage is a structured table")


class SourceCitation(BaseModel):
    source_file: str
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    page_number: Optional[int] = 1
    is_table: Optional[bool] = False
    source: Optional[str] = None
    text_excerpt: Optional[str] = None
    relevance_score: Optional[float] = None

    def model_post_init(self, __context):
        if not self.source:
            self.source = self.source_file
        if not self.text_excerpt:
            self.text_excerpt = self.snippet or ""
        if self.relevance_score is None:
            self.relevance_score = self.score or 0.0


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: Optional[int] = None
    chunks: Optional[int] = None
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None
    status: str = "indexed"

    def model_post_init(self, __context):
        if self.pages is None and self.total_pages is not None:
            self.pages = self.total_pages
        if self.total_pages is None and self.pages is not None:
            self.total_pages = self.pages
        if self.chunks is None and self.total_chunks is not None:
            self.chunks = self.total_chunks
        if self.total_chunks is None and self.chunks is not None:
            self.total_chunks = self.chunks


class QueryStreamRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1500, description="User search query or question")
    strategy: str = Field(default="rerank", description="Retrieval strategy: hybrid, rerank, simple, semantic, etc.")
    document_id: Optional[str] = Field(default=None, description="Optional document ID to scope retrieval")
    top_k: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = Field(default=None, description="LLM provider: groq, ollama, openai, anthropic")
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    mmr_lambda: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    domain_mode: Optional[str] = Field(default="general", description="Domain Persona: legal, finance, healthcare, tech, general")
    target_language: Optional[str] = Field(default=None, description="Target language: en, hi, es, de, fr, auto, etc.")
    air_gapped: Optional[bool] = Field(default=False, description="Whether 100% offline air-gapped privacy mode is active")


class AskRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    strategy: str = Field(default="rerank")
    document_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = None
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    mmr_lambda: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    session_id: Optional[str] = None
    domain_mode: Optional[str] = Field(default="general", description="Domain Persona: legal, finance, healthcare, tech, general")
    target_language: Optional[str] = Field(default=None, description="Target language: en, hi, es, de, fr, auto, etc.")
    air_gapped: Optional[bool] = Field(default=False, description="Whether 100% offline air-gapped privacy mode is active")

    def get_query(self) -> str:
        return self.query or self.question or ""


class LatencyBreakdown(BaseModel):
    embed_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    gen_ms: Optional[float] = None
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    citations: Optional[list[CitationItem]] = None
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
