"""
tests/test_privacy_guard.py — Unit tests for Air-Gapped mode enforcer
"""

import pytest
from config.privacy_guard import PrivacyGuard


def test_privacy_guard_air_gapped_enforcement():
    guard = PrivacyGuard(air_gapped_default=True)
    provider, is_offline = guard.validate_provider_request("groq")
    assert is_offline is True
    assert provider == "ollama"

    status = guard.get_compliance_status(air_gapped=True)
    assert status["air_gapped_active"] is True
    assert status["external_egress_allowed"] is False
    assert "GDPR" in status["status_banner"]


def test_privacy_guard_standard_mode():
    guard = PrivacyGuard(air_gapped_default=False)
    provider, is_offline = guard.validate_provider_request("groq", air_gapped=False)
    assert is_offline is False
    assert provider == "groq"
