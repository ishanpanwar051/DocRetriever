from typing import Optional
from sqlalchemy import text
from src.db.connection import get_db
from .base import Retriever, Chunk

class SemanticRetriever(Retriever):
    """
    Strategy 2: Semantic chunking + vector search.
    """
    def __init__(self, top_k=5, threshold=0.3):
        super().__init__(top_k)
        self.threshold = threshold
        self.chunk_strategy = 'semantic'
    
    def retrieve(self, query: str, document_id: Optional[str] = None) -> list[Chunk]:
        query_vec = self.embed_query(query)
        
        doc_filter = ""
        params = {
            "query_vec": str(query_vec),
            "chunk_strategy": self.chunk_strategy,
            "top_k": self.top_k,
        }
        
        if document_id:
            doc_filter = "AND (source_file = :doc_id OR metadata->>'document_id' = :doc_id OR metadata->>'filename' = :doc_id)"
            params["doc_id"] = document_id

        query_sql = text(f"""
            SELECT id, source_file, section_title, chunk_index, content, metadata,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity_score
            FROM document_chunks
            WHERE (chunk_strategy = :chunk_strategy OR :chunk_strategy = 'any' OR chunk_strategy IS NULL)
              {doc_filter}
            ORDER BY embedding <=> CAST(:query_vec AS vector) ASC
            LIMIT :top_k
        """)
        
        chunks = []
        with get_db() as db:
            result = db.execute(query_sql, params)
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
