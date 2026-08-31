from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


class LocalSentenceEmbedder:
    """
    Embeds text using local sentence-transformers models on CPU (e.g. all-MiniLM-L6-v2).
    WHY: batch_size=32 trade-off: Larger batch sizes reduce the number of forward passes
    (improving throughput), but smaller batch sizes keep peak memory tight. 32 is optimal
    for CPU inference under 8GB constraints.
    """
    def __init__(self, model='all-MiniLM-L6-v2', batch_size=32):
        self.model_name = model
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load the model to avoid loading at import time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts in batches.
        Uses sentence-transformers local model with normalization for exact cosine metrics.
        """
        if not texts:
            return []
            
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings.extend(batch_embeddings.tolist())
                      
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Embeds a single piece of text.
        WHY: Used for query embedding at retrieval time, optimizing for latency.
        """
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# Backward compatibility alias
OllamaEmbedder = LocalSentenceEmbedder
