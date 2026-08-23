from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings

@dataclass
class Chunk:
    id: int
    content: str
    source_file: str
    section_title: Optional[str]
    chunk_index: int
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

class Retriever(ABC):
    """
    Abstract base class for all 4 retrieval strategies.
    
    WHY ABSTRACT BASE CLASS: 
    - Ensures a unified interface (`retrieve`) across all retrieval methods. This allows downstream components
      (like generation or evaluation pipelines) to treat all strategies uniformly, facilitating easy swapping and fair comparison.
    - Promotes code reuse for common operations like `embed_query`.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    @abstractmethod
    def retrieve(self, query: str) -> list[Chunk]:
        """Retrieve top_k most relevant chunks for query."""
        pass
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed query using the configured Ollama model. Shared by all strategies.
        
        WHY SHARE THIS: Query embedding is universal across vector-based retrievers. 
        Centralizing it avoids duplication and ensures identical preprocessing.
        """
        import ollama
        resp = ollama.embed(model=settings.ollama_embed_model, input=query)
        if hasattr(resp, 'embeddings'):
            return resp.embeddings[0]
        return resp['embeddings'][0]
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
