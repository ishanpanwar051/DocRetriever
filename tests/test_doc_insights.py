"""
tests/test_doc_insights.py — Unit tests for Document Intelligence & Analytics Engine
"""

import pytest
from src.analytics.doc_insights import DocumentInsightsEngine


def test_document_insights_kpis_and_summary():
    engine = DocumentInsightsEngine()
    sample_text = (
        "In Q3 2024, Acme Technologies Corp generated $45.2 million in ARR with an operating margin of 18.5%. "
        "The vendor provides unlimited liability protection and guarantees GDPR and HIPAA compliance for all patient records. "
        "Under Section 14, material breach shall incur penalty fee of $500,000 payable to Chief Financial Officer within 30 days."
    )

    results = engine.analyze_document(sample_text, filename="acme_q3_report.pdf")

    assert results["filename"] == "acme_q3_report.pdf"
    assert results["kpis"]["total_words"] > 30
    assert results["kpis"]["reading_time_min"] > 0
    assert len(results["executive_summary"]) >= 2
    assert results["risk_summary"]["total_risks"] >= 1

    # Verify risk flags
    categories = [r["category"] for r in results["risk_flags"]]
    assert any("Liability" in c or "Breach" in c or "Compliance" in c for c in categories)

    # Verify entity extraction
    entities = results["entities"]
    types = [e["type"] for e in entities]
    assert "MONEY" in types or "DATE" in types or "ORG" in types
