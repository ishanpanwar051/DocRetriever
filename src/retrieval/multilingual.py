"""
src/retrieval/multilingual.py — Multilingual Ingestion & Cross-Lingual Querying Engine

Features:
- Automated Language Detection (Unicode script heuristics + langdetect fallback)
- Cross-Lingual Query Shaping & Response Prompting
- Native Support for English, Hindi, Spanish, German, French, Japanese, Chinese, etc.
"""

import re
from typing import Dict, Optional, Tuple

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
}


def detect_language(text: str) -> Tuple[str, str, float]:
    """
    Detects the primary language of the text.
    
    Returns:
        (lang_code, lang_name, confidence)
    """
    if not text or not text.strip():
        return "en", "English", 1.0

    clean_text = text.strip()

    # 1. Unicode Range Heuristics (High accuracy, Zero dependencies)
    # Devanagari (Hindi, Marathi, Sanskrit)
    if re.search(r"[\u0900-\u097F]", clean_text):
        return "hi", "Hindi", 0.98

    # Chinese (CJK Unified Ideographs)
    if re.search(r"[\u4E00-\u9FFF]", clean_text):
        return "zh", "Chinese", 0.98

    # Japanese (Hiragana / Katakana)
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", clean_text):
        return "ja", "Japanese", 0.98

    # Arabic
    if re.search(r"[\u0600-\u06FF]", clean_text):
        return "ar", "Arabic", 0.98

    # Cyrillic (Russian)
    if re.search(r"[\u0400-\u04FF]", clean_text):
        return "ru", "Russian", 0.98

    # 2. Try langdetect if installed
    try:
        from langdetect import detect_langs
        predictions = detect_langs(clean_text)
        if predictions:
            best = predictions[0]
            code = best.lang.lower()
            name = LANGUAGE_NAMES.get(code, code.title())
            return code, name, round(float(best.prob), 2)
    except Exception:
        pass

    # 3. Simple European Language Keyword Heuristics
    words = set(re.findall(r"\b\w+\b", clean_text.lower()))
    
    # Spanish
    if words & {"que", "como", "cual", "donde", "por", "para", "cuanto", "resumen", "informe"}:
        return "es", "Spanish", 0.85

    # German
    if words & {"was", "wie", "wo", "warum", "der", "die", "das", "und", "nicht", "bericht"}:
        return "de", "German", 0.85

    # French
    if words & {"qui", "que", "comment", "pourquoi", "dans", "avec", "pour", "rapport"}:
        return "fr", "French", 0.85

    # Hindi Hinglish (Romanized Hindi words)
    if words & {"kya", "kaise", "kitna", "batao", "munafa", "kamai", "fayda", "saar", "karo", "hai", "hain", "karna"}:
        return "hi", "Hindi (Hinglish)", 0.90

    return "en", "English", 0.95


def build_cross_lingual_instruction(
    query_text: str,
    target_language: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Analyzes query language and generates instruction for the LLM generator.
    
    Returns:
        (lang_code, lang_name, instruction_clause)
    """
    if target_language and target_language.lower() != "auto":
        lang_code = target_language.lower()
        lang_name = LANGUAGE_NAMES.get(lang_code, target_language.title())
    else:
        lang_code, lang_name, _ = detect_language(query_text)

    if lang_code == "en" and not target_language:
        instruction = ""
    else:
        instruction = (
            f"\n\n🌍 MULTILINGUAL ACCURACY MANDATE:\n"
            f"- The user's query is in {lang_name} (Code: {lang_code}).\n"
            f"- You MUST formulate your entire response in {lang_name}.\n"
            f"- Ground all factual statements, metrics, and numerical data strictly in the provided English/source context.\n"
            f"- Maintain all structured page citations, section titles, and Markdown tables intact."
        )

    return lang_code, lang_name, instruction
