from pydantic import BaseModel
from typing import Optional
import httpx
from config.settings import settings
from src.retrieval.base import Chunk
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE

class SourceCitation(BaseModel):
    source_file: str
    section_title: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    strategy: str
    num_context_chunks: int
    retrieval_scores: list[float]

class RAGGenerator:
    """
    Generates answers from retrieved context using Groq Cloud API.
    
    WHY structured output: Returning a Pydantic model ensures the API contracts are maintained
    and makes it easier to track exact sources and scores for evaluation and UI display.
    
    WHY temperature=0.2: A lower temperature increases the determinism of the LLM, making it 
    much less likely to hallucinate information outside of the provided context.
    
    WHY max_tokens=512: This provides enough tokens for a comprehensive answer while capping
    generation time and resource usage for overly verbose responses.
    """
    def __init__(self, model: str = None):
        self.model = model or settings.groq_llm_model
        self.temperature = 0.2
        self.max_tokens = 512
        self.api_key = settings.groq_api_key
        self.base_url = settings.groq_base_url.rstrip("/")
    
    def generate(self, question: str, chunks: list[Chunk], strategy: str) -> RAGResponse:
        # 1. Build context string from chunks
        context = self._build_context(chunks)
        
        # 2. Build prompt
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        
        # 3. Generate with Groq API (OpenAI-compatible)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        
        answer = data["choices"][0]["message"]["content"]
        
        # 4. Build structured response
        sources = [
            SourceCitation(
                source_file=c.source_file,
                section_title=c.section_title,
            )
            for c in chunks
        ]
        
        # Deduplicate sources while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.source_file, s.section_title)
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        
        return RAGResponse(
            answer=answer,
            sources=unique_sources,
            strategy=strategy,
            num_context_chunks=len(chunks),
            retrieval_scores=[c.score for c in chunks],
        )
    
    def _build_context(self, chunks: list[Chunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f'[{i}] Source: {chunk.source_file}'
            if chunk.section_title:
                header += f' | Section: {chunk.section_title}'
            parts.append(f'{header}\n{chunk.content}')
        return '\n\n---\n\n'.join(parts)
