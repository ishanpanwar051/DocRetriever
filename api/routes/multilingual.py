"""
api/routes/multilingual.py — Multilingual Ingestion & Cross-Lingual Querying Router
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from src.retrieval.multilingual import detect_language, LANGUAGE_NAMES

router = APIRouter(prefix="/api/multilingual", tags=["Multilingual Engine"])


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class DetectLanguageResponse(BaseModel):
    language_code: str
    language_name: str
    confidence: float


@router.post("/detect", response_model=DetectLanguageResponse)
def detect_lang(request: DetectLanguageRequest):
    """Detects the language of the provided text passage or query."""
    code, name, conf = detect_language(request.text)
    return DetectLanguageResponse(
        language_code=code,
        language_name=name,
        confidence=conf,
    )


@router.get("/languages")
def get_supported_languages() -> Dict[str, str]:
    """Returns the list of supported multilingual languages."""
    return LANGUAGE_NAMES
