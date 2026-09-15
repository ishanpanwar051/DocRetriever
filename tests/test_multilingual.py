"""
tests/test_multilingual.py — Unit tests for multilingual language detection and prompt shaping
"""

import pytest
from src.retrieval.multilingual import detect_language, build_cross_lingual_instruction


def test_detect_language_hindi():
    hindi_text = "कंपनी का कुल मुनाफा कितना था?"
    code, name, conf = detect_language(hindi_text)
    assert code == "hi"
    assert "Hindi" in name
    assert conf > 0.9


def test_detect_language_spanish():
    spanish_text = "¿Cuál es el resumen del informe financiero?"
    code, name, conf = detect_language(spanish_text)
    assert code == "es"
    assert "Spanish" in name


def test_detect_language_english():
    eng_text = "How does query parameter validation work with Pydantic in FastAPI?"
    code, name, conf = detect_language(eng_text)
    assert code == "en"
    assert name == "English"


def test_build_cross_lingual_instruction():
    hindi_query = "कंपनी का कुल मुनाफा कितना था?"
    code, name, instruction = build_cross_lingual_instruction(hindi_query)
    assert code == "hi"
    assert "Hindi" in instruction
    assert "MULTILINGUAL ACCURACY MANDATE" in instruction


def test_build_cross_lingual_instruction_explicit_target():
    eng_query = "What is the revenue?"
    code, name, instruction = build_cross_lingual_instruction(eng_query, target_language="es")
    assert code == "es"
    assert "Spanish" in instruction
