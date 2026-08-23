import gc
import logging
from config.settings import settings
from .base import Retriever, Chunk
from .simple import SimpleRetriever

logger = logging.getLogger(__name__)

class RerankRetriever(Retriever):
    """
    Strategy 4: Retrieve broad candidates then re-rank with cross-encoder.
    
    WHY CROSS-ENCODER VS BI-ENCODER:
    - Bi-encoder (what we do in strategies 1-3): encodes query and doc SEPARATELY,
      and we just compare their vectors. Fast, cacheable, but misses nuanced interactions
      between specific words in the query and the document.
    - Cross-encoder: The query AND document are concatenated and fed TOGETHER through
      the transformer layers. This models deep attention between query terms and doc terms,
      yielding much higher relevance accuracy.
      
    WHY THE 2-STAGE APPROACH:
    - Tradeoff: Cross-encoders are O(N) expensive because we must run inference at query time 
      for every single candidate. We cannot afford to score 10,000 documents this way.
    - Solution: Use a fast bi-encoder (SimpleRetriever) to find the top 20 candidates, 
      then use the expensive cross-encoder to accurately re-rank those 20 down to the top 5.
    - This architecture mirrors production systems (e.g., Cohere Rerank API).
    """
    _model = None  # Class-level cache ensures model is loaded only once per process
    
    def __init__(self, top_k=5, candidates=20, base_retriever=None):
        super().__init__(top_k)
        self.candidates = candidates
        self.base_retriever = base_retriever or SimpleRetriever(top_k=candidates)
    
    def _get_model(self):
        """
        Lazy load CrossEncoder. 
        Downloads model on first call (~0.3GB) and holds it in memory.
        """
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
        """
        Explicitly release the cross-encoder from RAM.
        WHY: On 8GB machines, holding bge-reranker-base (~0.3GB) permanently
        limits headroom for LLM generation. Call this when switching away from
        rerank strategy or before loading llama3.2:3b.
        """
        if cls._model is not None:
            logger.info("[RAM] Unloading bge-reranker-base from memory")
            del cls._model
            cls._model = None
            gc.collect()
    
    def retrieve(self, query: str) -> list[Chunk]:
        # Step 1: Retrieve a broader set of candidates quickly (e.g. top 20)
        candidates = self.base_retriever.retrieve(query)
        
        if not candidates:
            return []
        
        # Step 2: Prepare pairs and score with the cross-encoder
        model = self._get_model()
        pairs = [[query, c.content] for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        
        # Step 3: Re-rank based on the superior cross-encoder scores
        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = []
        for chunk, score in reranked[:self.top_k]:
            chunk.score = float(score)  # Replace bi-encoder score with cross-encoder score
            result.append(chunk)
        
        return result
