"""
src/ingestion/embedder.py — Embedding Providers for DocuMind

Provides:
- LocalSentenceEmbedder: Local CPU embedding via sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- OllamaEmbedder: Ollama /api/embeddings client (nomic-embed-text) with automatic failover to local model
"""

import httpx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config.settings import settings


class LocalSentenceEmbedder:
    """
    Embeds text using local sentence-transformers models on CPU (e.g. all-MiniLM-L6-v2).
    WHY: batch_size=32 trade-off: Optimal batch size for CPU throughput under memory constraints.
    """
    def __init__(self, model: str = None, batch_size: int = 32):
        self.model_name = model or settings.embed_model
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load the model to avoid loading at import time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings.extend(batch_embeddings.tolist())
                      
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


class OllamaEmbedder:
    """
    Embeds text via local Ollama instance (e.g. nomic-embed-text / all-minilm).
    Gracefully falls back to LocalSentenceEmbedder if Ollama is unreachable.
    """
    def __init__(self, model: str = None, base_url: str = None, batch_size: int = 32):
        self.model_name = model or getattr(settings, "ollama_embed_model", "nomic-embed-text")
        self.base_url = (base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.batch_size = batch_size
        self._fallback_embedder = None

    def _get_fallback(self) -> LocalSentenceEmbedder:
        if self._fallback_embedder is None:
            self._fallback_embedder = LocalSentenceEmbedder()
        return self._fallback_embedder

    def embed_single(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    emb = data.get("embedding", [])
                    if emb:
                        # If dimension matches or needs normalization
                        return emb
        except Exception:
            pass
        return self._get_fallback().embed_single(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Test connection first
        try:
            first_emb = self.embed_single(texts[0])
            embeddings = [first_emb]
            for t in texts[1:]:
                embeddings.append(self.embed_single(t))
            return embeddings
        except Exception:
            return self._get_fallback().embed_texts(texts)


def get_embedder(prefer_ollama: bool = False):
    """Factory helper to obtain the preferred embedder instance."""
    if prefer_ollama:
        return OllamaEmbedder()
    return LocalSentenceEmbedder()
