from pydantic import BaseModel
from typing import Optional
import ollama
from config.settings import settings
from src.retrieval.base import Chunk
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.utils.memory import OllamaModelManager

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
    Generates answers from retrieved context using llama3.2:3b.
    
    WHY structured output: Returning a Pydantic model ensures the API contracts are maintained
    and makes it easier to track exact sources and scores for evaluation and UI display.
    
    WHY temperature=0.2: A lower temperature increases the determinism of the LLM, making it 
    much less likely to hallucinate information outside of the provided context.
    
    WHY max_tokens=512: This provides enough tokens for a comprehensive answer while capping
    generation time and resource usage for overly verbose responses.
    """
    def __init__(self, model: str = None):
        self.model = model or settings.ollama_llm_model
        self.temperature = 0.2
        self.max_tokens = 512
    
    def generate(self, question: str, chunks: list[Chunk], strategy: str) -> RAGResponse:
        # 1. RAM: unload embedder, load LLM
        # WHY memory management: Only keeping one model in memory at a time prevents OOM errors
        # on resource-constrained environments (RAM Rule).
        mgr = OllamaModelManager(settings.ollama_base_url)
        
        # 2. Build context string from chunks
        context = self._build_context(chunks)
        
        # 3. Build prompt
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        
        # 4. Generate with Ollama
        resp = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_msg},
            ],
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
            },
            keep_alive=settings.ollama_keep_alive,  # 0 = unload after
        )
        
        answer = resp.message.content
        
        # 5. Build structured response
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
