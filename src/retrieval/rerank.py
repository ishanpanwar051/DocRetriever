from typing import Optional
import gc
import logging
from config.settings import settings
from .base import Retriever, Chunk
from .simple import SimpleRetriever

logger = logging.getLogger(__name__)

class RerankRetriever(Retriever):
    """
    Strategy 4: Retrieve broad candidates then re-rank with cross-encoder.
    """
    _model = None  # Class-level cache ensures model is loaded only once per process
    
    def __init__(self, top_k=5, candidates=20, base_retriever=None):
        super().__init__(top_k)
        self.candidates = candidates
        self.base_retriever = base_retriever or SimpleRetriever(top_k=candidates)
    
    def _get_model(self):
        if RerankRetriever._model is None:
            from sentence_transformers import CrossEncoder
            RerankRetriever._model = CrossEncoder(
                settings.reranker_model,
                device=settings.reranker_device,
                max_length=512,
            )
        return RerankRetriever._model
    
    @classmethod
    def unload_model(cls) -> None:
        if cls._model is not None:
            logger.info("[RAM] Unloading bge-reranker-base from memory")
            del cls._model
            cls._model = None
            gc.collect()
    
    def retrieve(self, query: str, document_id: Optional[str] = None) -> list[Chunk]:
        # Step 1: Retrieve a broader set of candidates quickly (e.g. top 20)
        candidates = self.base_retriever.retrieve(query, document_id=document_id)
        
        if not candidates:
            return []
        
        # Step 2: Prepare pairs and score with the cross-encoder
        try:
            model = self._get_model()
            pairs = [[query, c.content] for c in candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            
            # Step 3: Re-rank based on cross-encoder scores
            reranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            result = []
            for chunk, score in reranked[:self.top_k]:
                chunk.score = float(score)
                result.append(chunk)
            
            return result
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking fallback: {e}")
            return candidates[:self.top_k]
