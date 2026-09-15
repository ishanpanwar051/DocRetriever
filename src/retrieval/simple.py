from typing import Optional
import numpy as np
from sqlalchemy import text
from src.db.connection import get_db
from .base import Retriever, Chunk


class SimpleRetriever(Retriever):
    """
    Strategy 1: Simple vector similarity search with in-memory resilient fallback.
    """
    def __init__(self, top_k=5, chunk_strategy='simple'):
        super().__init__(top_k)
        self.chunk_strategy = chunk_strategy
    
    def retrieve(self, query: str, document_id: Optional[str] = None) -> list[Chunk]:
        query_vec = self.embed_query(query)
        from src.ingestion.ingest import MEMORY_DOCUMENTS_STORE
        
        # 1. Fast path: check in-memory cache if document_id is provided
        if document_id and (document_id in MEMORY_DOCUMENTS_STORE):
            cached = MEMORY_DOCUMENTS_STORE[document_id]
            q_emb = np.array(query_vec)
            scored = []
            for item in cached:
                c_emb = np.array(item["embedding"])
                score = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-8))
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                Chunk(
                    id=i,
                    content=it["content"],
                    source_file=it["source_file"],
                    section_title=it.get("section_title"),
                    chunk_index=it.get("chunk_index", i),
                    score=round(float(sc), 4),
                    metadata=it.get("metadata_", {})
                )
                for i, (sc, it) in enumerate(scored[:self.top_k], 1)
            ]

        # 2. Database search path
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
        try:
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
        except Exception:
            # Fallback across all in-memory chunks
            all_cached = []
            for k, doc_chunks in MEMORY_DOCUMENTS_STORE.items():
                all_cached.extend(doc_chunks)
            if all_cached:
                q_emb = np.array(query_vec)
                scored = []
                for item in all_cached:
                    c_emb = np.array(item["embedding"])
                    score = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-8))
                    scored.append((score, item))
                scored.sort(key=lambda x: x[0], reverse=True)
                for i, (sc, it) in enumerate(scored[:self.top_k], 1):
                    chunks.append(Chunk(
                        id=i,
                        content=it["content"],
                        source_file=it["source_file"],
                        section_title=it.get("section_title"),
                        chunk_index=it.get("chunk_index", i),
                        score=round(float(sc), 4),
                        metadata=it.get("metadata_", {})
                    ))

        return chunks
