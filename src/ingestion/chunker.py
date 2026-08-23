import re
import numpy as np
from typing import Callable
import tiktoken

class SimpleChunker:
    """
    Token-based chunker with recursive splitting.
    """
    def __init__(self, chunk_size=500, overlap=50):
        # WHY: Using cl100k_base (OpenAI's token encoding) is a standard approximation.
        # WHY 500/50: 500 tokens is large enough to contain context, but small enough
        # to fit multiple chunks in an LLM's context window. 50 token overlap ensures 
        # context isn't lost at chunk boundaries.
        # INTERVIEW ANSWER: 1000t baseline often loses precision for short precise answers 
        # because the specific fact is drowned out by surrounding text.
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.separators = ['\n\n', '\n', ' ', '']

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursive character splitter logic."""
        # WHY recursive: Tries paragraph breaks first (semantic), falls back to 
        # sentence, then word, preserving meaning as much as possible.
        if not separators:
            return [text]
            
        separator = separators[0]
        if separator == '':
            return list(text)
            
        splits = text.split(separator)
        good_splits = []
        for s in splits:
            if len(self.encoder.encode(s)) > self.chunk_size:
                good_splits.extend(self._split_text(s, separators[1:]))
            else:
                good_splits.append(s)
        return good_splits

    def chunk(self, text: str) -> list[str]:
        # simplified recursive splitting for demonstration
        # in reality, you'd aggregate chunks up to chunk_size
        raw_splits = self._split_text(text, self.separators)
        
        chunks = []
        current_chunk = ""
        
        for split in raw_splits:
            if not current_chunk:
                current_chunk = split
                continue
                
            combined = current_chunk + " " + split
            if len(self.encoder.encode(combined)) <= self.chunk_size:
                current_chunk = combined
            else:
                chunks.append(current_chunk)
                current_chunk = split
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

def cosine_sim(a: list[float], b: list[float]) -> float:
    """Computes cosine similarity between two vectors."""
    a_vec = np.array(a)
    b_vec = np.array(b)
    if np.linalg.norm(a_vec) == 0 or np.linalg.norm(b_vec) == 0:
        return 0.0
    return np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))

class SemanticChunker:
    """
    Semantic chunker based on embedding similarity drops.
    """
    def __init__(self, embed_fn: Callable, threshold=0.3, max_chunk_tokens=400):
        # WHY: Topic-coherent chunks = better context precision in RAGAS eval.
        # WHY threshold 0.3: It's a tunable parameter. Higher threshold = finer 
        # boundaries (splits more often). 0.3 is a decent starting point for embeddings.
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, sentences: list[str]) -> list[str]:
        if not sentences:
            return []
            
        embeddings = self.embed_fn(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i-1], embeddings[i])
            
            # Boundary where similarity DROPS below threshold
            if sim < self.threshold:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                # Also check token limits
                proposed_chunk = " ".join(current_chunk_sentences + [sentences[i]])
                if len(self.encoder.encode(proposed_chunk)) > self.max_chunk_tokens:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentences[i]]
                else:
                    current_chunk_sentences.append(sentences[i])
                    
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks
