"""
src/generation/prompts.py — Grounded Prompt Templates with Domain Persona & Multilingual Shaping
"""

from config.domain_profiles import get_domain_prompt
from src.retrieval.multilingual import build_cross_lingual_instruction

BASE_SYSTEM_PROMPT = """You are a precise, enterprise-grade AI documentation assistant.
Answer questions ONLY using the provided context passages.
If the answer is not in the context, respond with exactly: "I cannot find this information in the provided context."
Always cite your sources and page numbers clearly.

Few-shot example of BAD answer (do not do this):
Question: What is the default port for FastAPI?
Context: [passage about routing]
BAD: The default port is 8000. (fabricated - not in context!)

Correcting example:
Good: "I cannot find this information in the provided context."
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer based ONLY on the context above. Cite source files and page numbers.
If context doesn't contain the answer, say "I cannot find this information in the provided context."
"""


def build_system_prompt(
    domain_mode: str = "general",
    target_language: str = None,
    query_text: str = ""
) -> str:
    """
    Constructs a customized system prompt based on industry domain persona
    and query language detection.
    """
    base = get_domain_prompt(domain_mode, target_language=target_language)
    
    # Add cross-lingual mandate if query is in non-English
    if query_text:
        _, _, cross_inst = build_cross_lingual_instruction(query_text, target_language=target_language)
        if cross_inst:
            base += f"\n{cross_inst}"

    return base
