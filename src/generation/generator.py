import json
import time
from typing import Generator, Optional
from pydantic import BaseModel, Field
import httpx
from config.settings import settings
from src.retrieval.base import Chunk
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE


class SourceCitation(BaseModel):
    source_file: str
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    strategy: str
    num_context_chunks: int
    retrieval_scores: list[float]
    grounded: bool = True
    grounding_note: Optional[str] = None
    latency_ms: Optional[float] = None
    provider_used: Optional[str] = None


class LLMProvider:
    """Base provider interface for multi-provider generation."""
    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
        raise NotImplementedError

    def generate_stream(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> Generator[str, None, None]:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.model = model or settings.groq_llm_model

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate_stream(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> Generator[str, None, None]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.model = model or getattr(settings, "ollama_llm_model", "llama3.2:3b")

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": False}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")

    def generate_stream(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": True}
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            d = json.loads(line)
                            content = d.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue


def get_llm_provider(provider_name: str = None) -> LLMProvider:
    """
    Returns active LLM provider with graceful failover:
    1. If requested provider is available, use it.
    2. If Groq has an API key, default to Groq.
    3. If Groq unavailable/no key, fallback to local Ollama.
    """
    name = (provider_name or settings.default_llm_provider).lower()
    if name == "groq" and settings.groq_api_key:
        return GroqProvider()
    elif name == "ollama":
        return OllamaProvider()
    elif settings.groq_api_key:
        return GroqProvider()
    else:
        return OllamaProvider()


class RAGGenerator:
    """
    Multi-Provider RAG Generator supporting synchronous response building and token streaming.
    """
    def __init__(self, provider: str = None, model: str = None):
        self.provider_name = provider or settings.default_llm_provider
        self.provider = get_llm_provider(self.provider_name)
        self.temperature = 0.2
        self.max_tokens = 512

    def generate(self, question: str, chunks: list[Chunk], strategy: str) -> RAGResponse:
        start_time = time.time()
        context = self._build_context(chunks)
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            answer = self.provider.generate(messages, temperature=self.temperature, max_tokens=self.max_tokens)
            provider_used = self.provider.__class__.__name__.replace("Provider", "").lower()
        except Exception as e:
            # Automatic Failover to Ollama or Fallback
            try:
                fallback_p = OllamaProvider()
                answer = fallback_p.generate(messages, temperature=self.temperature, max_tokens=self.max_tokens)
                provider_used = "ollama (failover)"
            except Exception:
                answer = f"I could not reach the LLM generator. Details: {e}"
                provider_used = "error"

        duration_ms = (time.time() - start_time) * 1000

        # Grounding & Citations
        sources = [
            SourceCitation(
                source_file=c.source_file,
                section_title=c.section_title,
                snippet=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                score=c.score,
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

        # Grounding verification check
        is_refusal = "cannot find this information" in answer.lower()
        grounded = is_refusal or (len(chunks) > 0 and any(s.score is None or s.score > 0.005 for s in unique_sources))
        grounding_note = None if grounded else "⚠️ Low retrieval confidence: response may not be fully grounded in documentation."

        return RAGResponse(
            answer=answer,
            sources=unique_sources,
            strategy=strategy,
            num_context_chunks=len(chunks),
            retrieval_scores=[c.score for c in chunks],
            grounded=grounded,
            grounding_note=grounding_note,
            latency_ms=round(duration_ms, 2),
            provider_used=provider_used,
        )

    def generate_stream(self, question: str, chunks: list[Chunk], strategy: str) -> Generator[dict, None, None]:
        """
        Yields token events and final citation payload for Server-Sent Events (SSE).
        """
        context = self._build_context(chunks)
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # 1. Yield source metadata event first
        sources_payload = [
            {
                "source_file": c.source_file,
                "section_title": c.section_title,
                "snippet": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "score": c.score,
            }
            for c in chunks
        ]
        yield {"type": "metadata", "strategy": strategy, "sources": sources_payload, "num_chunks": len(chunks)}

        # 2. Stream tokens incrementally
        full_text = []
        try:
            for token in self.provider.generate_stream(messages, temperature=self.temperature, max_tokens=self.max_tokens):
                full_text.append(token)
                yield {"type": "token", "content": token}
        except Exception:
            # Fallback one-shot
            resp = self.generate(question, chunks, strategy)
            yield {"type": "token", "content": resp.answer}

        yield {"type": "done", "full_answer": "".join(full_text)}

    def _build_context(self, chunks: list[Chunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f'[{i}] Source: {chunk.source_file}'
            if chunk.section_title:
                header += f' | Section: {chunk.section_title}'
            parts.append(f'{header}\n{chunk.content}')
        return '\n\n---\n\n'.join(parts)

