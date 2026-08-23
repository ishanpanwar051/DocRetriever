from collections import defaultdict
from sqlalchemy import text
from src.db.connection import get_db
from .base import Retriever, Chunk

class HybridRetriever(Retriever):
    """
    Strategy 3: Hybrid search = vector (semantic) + keyword (BM25-like tsvector).
    
    WHY HYBRID:
    - Dense vectors are great for semantic meaning but struggle with exact matches (IDs, names, jargon).
    - Sparse keyword search (tsvector) is perfect for exact terminology but misses synonyms.
    - Combining them gives the best of both worlds.
    
    WHY RRF (Reciprocal Rank Fusion):
    - RRF score = sum(1/(k + rank_i)). It merges ranked lists without requiring the scores to be calibrated.
    - k=60 is empirically optimal (Cormack 2009) as it dampens the impact of extreme top ranks, smoothing the distribution.
    
    WHY ALPHA WEIGHTS:
    - Allows tuning the balance between semantic and keyword importance (e.g., alpha=0.7 favors semantic meaning).
    """
    def __init__(self, top_k=5, alpha=0.5, rrf_k=60, chunk_strategy='simple'):
        super().__init__(top_k)
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.chunk_strategy = chunk_strategy
    
    def retrieve(self, query: str) -> list[Chunk]:
        # Step 1: Vector search for broad semantic matches
        vector_results = self._vector_search(query, limit=20)
        
        # Step 2: Keyword search for exact terminology matches
        keyword_results = self._keyword_search(query, limit=20)
        
        # Step 3: RRF fusion to balance both sets
        fused = self._rrf_fusion(vector_results, keyword_results)
        
        return fused[:self.top_k]
    
    def _vector_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_vec = self.embed_query(query)
        sql = text("""
            SELECT id, 1 - (embedding <=> CAST(:query_vec AS vector)) AS score
            FROM document_chunks
            WHERE chunk_strategy = :chunk_strategy
            ORDER BY embedding <=> CAST(:query_vec AS vector) ASC
            LIMIT :limit
        """)
        with get_db() as db:
            result = db.execute(sql, {"query_vec": str(query_vec), "chunk_strategy": self.chunk_strategy, "limit": limit})
            return [(row.id, float(row.score)) for row in result]
            
    def _keyword_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        sql = text("""
            SELECT id, ts_rank(content_tsv, plainto_tsquery('english', :query)) AS score
            FROM document_chunks
            WHERE chunk_strategy = :chunk_strategy
              AND content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
        """)
        with get_db() as db:
            result = db.execute(sql, {"query": query, "chunk_strategy": self.chunk_strategy, "limit": limit})
            return [(row.id, float(row.score)) for row in result]
            
    def _rrf_fusion(self, vec_results: list[tuple[int, float]], kw_results: list[tuple[int, float]]) -> list[Chunk]:
        rrf_scores = defaultdict(float)
        
        for rank, (chunk_id, _) in enumerate(vec_results, start=1):
            rrf_scores[chunk_id] += self.alpha * (1.0 / (self.rrf_k + rank))
            
        for rank, (chunk_id, _) in enumerate(kw_results, start=1):
            rrf_scores[chunk_id] += (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank))
            
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in sorted_ids[:self.top_k]]
        
        if not top_ids:
            return []
            
        sql = text("""
            SELECT id, source_file, section_title, chunk_index, content, metadata
            FROM document_chunks
            WHERE id = ANY(:ids)
        """)
        
        chunks = []
        with get_db() as db:
            result = db.execute(sql, {"ids": top_ids})
            chunk_map = {
                row.id: Chunk(
                    id=row.id, content=row.content, source_file=row.source_file,
                    section_title=row.section_title, chunk_index=row.chunk_index,
                    metadata=row.metadata or {}
                ) for row in result
            }
            
            for cid, score in sorted_ids[:self.top_k]:
                if cid in chunk_map:
                    chunk = chunk_map[cid]
                    chunk.score = score
                    chunks.append(chunk)
                    
        return chunks
