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

    @property
    def page_number(self) -> int:
        """Extracts 1-indexed page number from metadata or section title fallback."""
        if self.metadata and "page_number" in self.metadata:
            try:
                return int(self.metadata["page_number"])
            except (ValueError, TypeError):
                pass
        # Fallback to section_title regex if available (e.g. "Page 3")
        if self.section_title:
            import re
            m = re.search(r"Page\s+(\d+)", self.section_title, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return 1

    @property
    def is_table(self) -> bool:
        """Determines if this chunk contains a preserved table."""
        if self.metadata and "is_table" in self.metadata:
            return bool(self.metadata["is_table"])
        return bool(self.section_title and "Table" in self.section_title)

    @property
    def document_id(self) -> Optional[str]:
        """Returns document identifier if present."""
        if self.metadata and "document_id" in self.metadata:
            return str(self.metadata["document_id"])
        return None

    @property
    def filename(self) -> str:
        """Returns source filename."""
        if self.metadata and "filename" in self.metadata:
            return str(self.metadata["filename"])
        return self.source_file

class Retriever(ABC):
    """
    Abstract base class for all retrieval strategies.
    Ensures a unified interface across dense, sparse, hybrid, and re-ranked methods.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._embedder = None
    
    @abstractmethod
    def retrieve(self, query: str, document_id: Optional[str] = None) -> list[Chunk]:
        """Retrieve top_k most relevant chunks for query, with optional document_id scope."""
        pass
    
    def embed_query(self, query: str) -> list[float]:
        """Embed query using sentence-transformers (local CPU model)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(settings.embed_model)
        embedding = self._embedder.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return embedding[0].tolist()
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
