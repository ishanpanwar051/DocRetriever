from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, Integer, TIMESTAMP, func, Index
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime
from config.settings import settings

class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy ORM models.
    WHY: DeclarativeBase in SQLAlchemy 2.0 is preferred over declarative_base() 
    because it provides stronger type hinting and better IDE support, which is 
    critical in a production codebase to prevent type-related bugs.
    """
    pass

class DocumentChunk(Base):
    """
    Represents a chunk of text from a document, along with its vector embedding.
    WHY: Storing the source file, section title, and chunk index allows us to 
    reconstruct the original document and provide precise provenance in RAG citations.
    """
    __tablename__ = 'document_chunks'
    
    # WHY: Integer primary key is efficient for indexing and foreign keys.
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # WHY: Storing the source file path helps in filtering by document.
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    
    # WHY: Optional section title for semantic context during retrieval.
    section_title: Mapped[str | None] = mapped_column(Text)
    
    # WHY: Chunk index helps to keep the ordering of chunks from the same document.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # WHY: Text column for the actual chunk content.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # WHY: Token count is useful for LLM context window management and cost estimation.
    token_count: Mapped[int | None] = mapped_column(Integer)
    
    # WHY: Dynamically configured vector dimension (384 for all-MiniLM-L6-v2)
    embedding = mapped_column(Vector(settings.embedding_dim), nullable=True)
    
    # WHY: Tracking chunk strategy (e.g., 'simple', 'semantic') allows A/B testing 
    # of different retrieval strategies.
    chunk_strategy: Mapped[str] = mapped_column(Text, default='simple')
    
    # WHY: JSONB is used for flexible metadata storage (e.g., tags, authors) 
    # which can be indexed in Postgres for fast filtering.
    metadata_: Mapped[dict] = mapped_column('metadata', JSONB, default=dict)
    
    # WHY: Timestamping helps in incremental updates and data lifecycle management.
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class EvalRun(Base):
    """
    Stores metrics and parameters for evaluation runs.
    WHY: Tracking evaluation metrics in the DB allows historical comparison of 
    retrieval strategies over time.
    """
    __tablename__ = 'eval_runs'
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class Document(Base):
    """
    Represents an ingested document with SHA-256 hash for incremental ingestion diffing.
    """
    __tablename__ = 'documents'
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_file: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    file_type: Mapped[str] = mapped_column(Text, default="markdown")
    file_hash: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    corpus_name: Mapped[str] = mapped_column(Text, default="default")
    metadata_: Mapped[dict] = mapped_column('metadata', JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class QueryLog(Base):
    """
    Captures stage-by-stage latencies, strategies, and generated answers for telemetry.
    """
    __tablename__ = 'query_logs'
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str | None] = mapped_column(Text)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    total_latency_ms: Mapped[float] = mapped_column(nullable=False)
    embed_latency_ms: Mapped[float | None] = mapped_column()
    retrieval_latency_ms: Mapped[float | None] = mapped_column()
    rerank_latency_ms: Mapped[float | None] = mapped_column()
    gen_latency_ms: Mapped[float | None] = mapped_column()
    sources: Mapped[list] = mapped_column(JSONB, default=list)
    answer: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class UserFeedback(Base):
    """
    Captures user feedback (thumbs up / down) on retrieved context and generated answers.
    """
    __tablename__ = 'user_feedback'
    id: Mapped[int] = mapped_column(primary_key=True)
    query_id: Mapped[int | None] = mapped_column(Integer)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = thumbs up, -1 = thumbs down
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
