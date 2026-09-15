"""
tests/test_voice.py — Unit tests for Voice-to-Voice synthesis and transcription endpoints
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_voice_synthesize_endpoint():
    response = client.post(
        "/api/voice/synthesize",
        json={"text": "Hello, welcome to DocuMind Enterprise.", "language": "en"},
    )
    assert response.status_code == 200
    assert "audio/" in response.headers.get("content-type", "")
    assert len(response.content) > 100


def test_multilingual_detect_endpoint():
    response = client.post(
        "/api/multilingual/detect",
        json={"text": "कंपनी का कुल मुनाफा कितना था?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language_code"] == "hi"
    assert "Hindi" in data["language_name"]


def test_document_insights_endpoint():
    response = client.post(
        "/api/documents/insights",
        json={
            "text_content": "Acme Corp earned $10M in revenue in 2024. All services are provided as is without warranty under GDPR compliance."
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert "executive_summary" in data
    assert "risk_flags" in data
