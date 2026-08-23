from sqlalchemy import text
from src.db.connection import get_db
from .base import Retriever, Chunk

class SemanticRetriever(Retriever):
    """
    Strategy 2: Semantic chunking + vector search.
    
    WHY DIFFERENT FROM SIMPLE:
    - Standard chunking (Strategy 1) blindly cuts text at fixed lengths, potentially splitting thoughts.
    - Semantic chunking dynamically splits text based on topic shifts (embedding distances between sentences).
    
    The retrieval mechanism is identical to SimpleRetriever (cosine similarity).
    The DIFFERENCE is in the underlying data: we query chunks that were created via semantic splitting.
    
    WHY THIS MATTERS FOR EVALUATION:
    - Allows us to isolate the effect of the chunking strategy on retrieval quality, holding the search algorithm constant.
    """
    def __init__(self, top_k=5, threshold=0.3):
        super().__init__(top_k)
        self.threshold = threshold  # Stored as metadata for experiment tracking
        self.chunk_strategy = 'semantic'
    
    def retrieve(self, query: str) -> list[Chunk]:
        query_vec = self.embed_query(query)
        
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
