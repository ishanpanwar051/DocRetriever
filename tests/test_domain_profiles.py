"""
tests/test_domain_profiles.py — Unit tests for industry domain personas
"""

import pytest
from config.domain_profiles import DOMAIN_PROFILES, get_domain_profile, get_domain_prompt


def test_domain_profiles_exist():
    assert "legal" in DOMAIN_PROFILES
    assert "finance" in DOMAIN_PROFILES
    assert "healthcare" in DOMAIN_PROFILES
    assert "tech" in DOMAIN_PROFILES
    assert "general" in DOMAIN_PROFILES


def test_get_domain_profile_legal():
    profile = get_domain_profile("legal")
    assert profile["id"] == "legal"
    assert "Legal Counsel" in profile["name"]
    assert "indemnif" in profile["system_prompt"].lower()


def test_get_domain_profile_finance():
    profile = get_domain_profile("finance")
    assert profile["id"] == "finance"
    assert "EBITDA" in profile["system_prompt"]
    assert "Markdown table" in profile["system_prompt"]


def test_get_domain_prompt_with_multilingual():
    prompt = get_domain_prompt("legal", target_language="hi")
    assert "Legal" in prompt
    assert "MULTILINGUAL INSTRUCTION" in prompt
    assert "hi" in prompt
