from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class OllamaEmbedder:
    def __init__(self, model='all-MiniLM-L6-v2', batch_size=32):
        """
        Embeds text using sentence-transformers (runs on CPU, no GPU needed).
        WHY: batch_size=32 trade-off: Larger batch sizes reduce the number of API calls 
        (improving throughput), but smaller batch sizes reduce peak RAM usage. 32 is a 
        sweet spot for avoiding OOM errors on 8GB machines while remaining efficient.
        """
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
        Uses sentence-transformers local model - no Ollama dependency.
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
                      
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Embeds a single piece of text.
        WHY: Used for query embedding at retrieval time, optimizing for latency.
        """
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()
