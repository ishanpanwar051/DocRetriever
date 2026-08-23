from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, Integer, TIMESTAMP, func, Index
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime

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
    
    # WHY: 768 is the embedding dimension for 'nomic-embed-text'.
    embedding = mapped_column(Vector(768), nullable=True)
    
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
