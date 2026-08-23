from sqlalchemy import text
from src.db.connection import get_db
from .base import Retriever, Chunk

class SimpleRetriever(Retriever):
    """
    Strategy 1: Simple vector similarity search.
    
    WHY THIS IS BASELINE:
    - Provides a standard dense retrieval implementation to benchmark more complex strategies against.
    
    WHY COSINE OVER L2:
    - Cosine similarity measures the angle between vectors, ignoring magnitude. In embedding spaces, 
      semantic similarity is usually represented by direction rather than magnitude. L2 distance can be
      skewed by vector length, making cosine (or inner product for normalized vectors) the standard choice.
    """
    def __init__(self, top_k=5, chunk_strategy='simple'):
        super().__init__(top_k)
        self.chunk_strategy = chunk_strategy
    
    def retrieve(self, query: str) -> list[Chunk]:
        query_vec = self.embed_query(query)
        
        # WHY RAW SQL WITH PSYCOPG2/SQLALCHEMY TEXT:
        # - Direct execution of pgvector's `<=>` operator (cosine distance) is highly optimized in the database.
        # - Avoiding ORM overhead for millions of vectors is crucial for latency. We only map to objects at the end.
        query_sql = text("""
            SELECT id, source_file, section_title, chunk_index, content, metadata,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity_score
            FROM document_chunks
            WHERE chunk_strategy = :chunk_strategy
            ORDER BY embedding <=> CAST(:query_vec AS vector) ASC
            LIMIT :top_k
        """)
        
        chunks = []
        with get_db() as db:
            result = db.execute(
                query_sql,
                {
                    "query_vec": str(query_vec),
                    "chunk_strategy": self.chunk_strategy,
                    "top_k": self.top_k
                }
            )
            for row in result:
                chunks.append(Chunk(
                    id=row.id,
                    content=row.content,
                    source_file=row.source_file,
                    section_title=row.section_title,
                    chunk_index=row.chunk_index,
                    score=float(row.similarity_score),
                    metadata=row.metadata or {}
                ))
        return chunks
