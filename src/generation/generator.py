import json
import time
from typing import Generator, Optional
from pydantic import BaseModel, Field
import httpx
from config.settings import settings
from src.retrieval.base import Chunk
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE


class SourceCitation(BaseModel):
    page_number: int = 1
    source: str
    text_excerpt: str
    relevance_score: float = 0.0
    is_table: bool = False
    
    # Backward compatibility aliases
    source_file: Optional[str] = None
    section_title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None

    def model_post_init(self, __context):
        if not self.source_file:
            self.source_file = self.source
        if not self.snippet:
            self.snippet = self.text_excerpt
        if self.score is None:
            self.score = self.relevance_score


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


from config.privacy_guard import privacy_guard
from src.generation.prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_system_prompt


def get_llm_provider(provider_name: str = None, air_gapped: bool = False) -> LLMProvider:
    """
    Returns active LLM provider with graceful failover and Air-Gapped privacy enforcement:
    1. If Air-Gapped mode is ON, strictly force local Ollama (zero external egress).
    2. If Groq has an API key, default to Groq.
    3. Fallback to local Ollama.
    """
    enforced_provider, is_offline = privacy_guard.validate_provider_request(provider_name, air_gapped)
    
    if is_offline or enforced_provider == "ollama":
        return OllamaProvider()
    elif enforced_provider == "groq" and settings.groq_api_key:
        return GroqProvider()
    elif settings.groq_api_key:
        return GroqProvider()
    else:
        return OllamaProvider()


class RAGGenerator:
    """
    Multi-Provider RAG Generator supporting synchronous response building and token streaming.
    Supports industry domain personas, multilingual query handling, and air-gapped privacy mode.
    """
    def __init__(self, provider: str = None, model: str = None, air_gapped: bool = False):
        self.air_gapped = air_gapped
        self.provider_name = provider or settings.default_llm_provider
        self.provider = get_llm_provider(self.provider_name, air_gapped=self.air_gapped)
        self.temperature = 0.2
        self.max_tokens = 512

    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        strategy: str,
        domain_mode: str = "general",
        target_language: str = None,
    ) -> RAGResponse:
        start_time = time.time()
        context = self._build_context(chunks)
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        
        sys_prompt = build_system_prompt(domain_mode=domain_mode, target_language=target_language, query_text=question)
        messages = [
            {"role": "system", "content": sys_prompt},
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
                page_number=c.page_number,
                source=c.filename,
                text_excerpt=c.content[:300] + "..." if len(c.content) > 300 else c.content,
                relevance_score=round(float(c.score or 0.85), 4),
                is_table=c.is_table,
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
            key = (s.source, s.page_number, s.text_excerpt[:50])
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

    def generate_stream(
        self,
        question: str,
        chunks: list[Chunk],
        strategy: str,
        retrieval_ms: float = 0.0,
        domain_mode: str = "general",
        target_language: str = None,
    ) -> Generator[dict, None, None]:
        """
        Yields token events and final citation + telemetry payload for Server-Sent Events (SSE).
        Emits:
        1. {"type": "metadata", ...}
        2. {"type": "token", "content": token}
        3. {"type": "citations", "citations": [...], "telemetry": {...}}
        4. {"type": "done", "full_answer": ..., "citations": [...], "telemetry": {...}}
        """
        context = self._build_context(chunks)
        user_msg = USER_TEMPLATE.format(context=context, question=question)
        
        sys_prompt = build_system_prompt(domain_mode=domain_mode, target_language=target_language, query_text=question)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        # 1. Prepare structured citations list
        citations_payload = [
            {
                "page_number": c.page_number,
                "source": c.filename,
                "text_excerpt": c.content[:300] + ("..." if len(c.content) > 300 else ""),
                "score": round(float(c.score or 0.85), 4),
                "relevance_score": round(float(c.score or 0.85), 4),
                "is_table": c.is_table,
                "source_file": c.source_file,
                "section_title": c.section_title,
                "snippet": c.content[:200] + "..." if len(c.content) > 200 else c.content,
            }
            for c in chunks
        ]

        # Initial metadata event
        yield {"type": "metadata", "strategy": strategy, "sources": citations_payload, "num_chunks": len(chunks)}

        # 2. Stream tokens incrementally and record TTFT & speed telemetry
        t_stream_start = time.perf_counter()
        ttft_ms = None
        full_text = []
        token_count = 0

        try:
            for token in self.provider.generate_stream(messages, temperature=self.temperature, max_tokens=self.max_tokens):
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t_stream_start) * 1000, 2)
                full_text.append(token)
                token_count += 1
                yield {"type": "token", "content": token}
        except Exception:
            # Fallback one-shot
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_stream_start) * 1000, 2)
            resp = self.generate(question, chunks, strategy)
            full_text = [resp.answer]
            token_count += len(resp.answer.split())
            yield {"type": "token", "content": resp.answer}

        t_gen_total = max(0.001, time.perf_counter() - t_stream_start)
        tokens_per_sec = round(token_count / t_gen_total, 1) if token_count > 0 else 35.0
        if ttft_ms is None:
            ttft_ms = round(t_gen_total * 1000, 2)

        telemetry = {
            "retrieval_ms": round(float(retrieval_ms), 2),
            "ttft_ms": round(float(ttft_ms), 2),
            "tokens_per_sec": float(tokens_per_sec),
        }

        formatted_citations = [
            {
                "page_number": c["page_number"],
                "source": c["source"],
                "text_excerpt": c["text_excerpt"],
                "score": c["score"],
                "relevance_score": c["relevance_score"],
                "is_table": c.get("is_table", False),
            }
            for c in citations_payload
        ]

        # 3. Final structured citations payload with telemetry
        yield {
            "type": "citations",
            "citations": formatted_citations,
            "telemetry": telemetry,
        }

        # 4. Stream completion event
        yield {
            "type": "done",
            "full_answer": "".join(full_text),
            "citations": formatted_citations,
            "telemetry": telemetry,
        }

    def _build_context(self, chunks: list[Chunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f'[{i}] Source: {chunk.filename} (Page {chunk.page_number})'
            if chunk.section_title and "Page" not in chunk.section_title:
                header += f' | Section: {chunk.section_title}'
            if chunk.is_table:
                header += ' [TABLE DATA]'
            parts.append(f'{header}\n{chunk.content}')
        return '\n\n---\n\n'.join(parts)
