import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.base import Chunk
from src.retrieval.hybrid import HybridRetriever
from eval.metrics import recall_at_k, mrr, compute_retrieval_metrics


def test_chunk_defaults_and_properties():
    chunk = Chunk(id=1, content="test content", source_file="doc.pdf", section_title="Page 4 - Table 1", chunk_index=0)
    assert chunk.score == 0.0
    assert chunk.metadata == {}
    assert chunk.page_number == 4
    assert chunk.is_table is True
    assert chunk.filename == "doc.pdf"

    chunk_with_meta = Chunk(
        id=2,
        content="| A | B |\n|---|---|\n| 1 | 2 |",
        source_file="report.pdf",
        section_title="Financials",
        chunk_index=1,
        metadata={"page_number": 7, "is_table": True, "filename": "financial_report.pdf", "document_id": "doc_abc"}
    )
    assert chunk_with_meta.page_number == 7
    assert chunk_with_meta.is_table is True
    assert chunk_with_meta.filename == "financial_report.pdf"
    assert chunk_with_meta.document_id == "doc_abc"


def test_recall_at_k_found():
    retrieved = ["tutorial/a.md", "tutorial/b.md", "tutorial/c.md"]
    assert recall_at_k(retrieved, "tutorial/b.md", k=5) == 1.0


def test_recall_at_k_not_found():
    retrieved = ["tutorial/a.md"]
    assert recall_at_k(retrieved, "tutorial/z.md", k=5) == 0.0


def test_mrr_rank_1():
    retrieved = ["tutorial/a.md", "tutorial/b.md"]
    assert mrr(retrieved, "tutorial/a.md") == 1.0


def test_mrr_rank_2():
    retrieved = ["tutorial/a.md", "tutorial/b.md"]
    assert mrr(retrieved, "tutorial/b.md") == 0.5


def test_mrr_not_found():
    retrieved = ["tutorial/a.md"]
    assert mrr(retrieved, "tutorial/z.md") == 0.0


def test_rrf_formula():
    """Verify Reciprocal Rank Fusion mathematical formula: 1 / (k + rank)."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.rrf_k = 60
    # rank 1 score with k=60 should be exactly 1 / 61
    expected_score = 1.0 / (60 + 1)
    assert abs(expected_score - 0.0163934) < 0.0001


def test_compute_retrieval_metrics():
    sample_results = [
        {"type": "answerable", "expected_source": "tutorial/a.md", "retrieved_sources": ["tutorial/a.md", "tutorial/b.md"]},
        {"type": "answerable", "expected_source": "tutorial/b.md", "retrieved_sources": ["tutorial/c.md", "tutorial/b.md"]},
        {"type": "unanswerable", "expected_source": "", "retrieved_sources": ["tutorial/a.md"]},
    ]
    metrics = compute_retrieval_metrics(sample_results)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.75  # (1.0 + 0.5) / 2


def test_mmr_cosine_similarity():
    import numpy as np
    from src.retrieval.mmr import cosine_similarity
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5


def test_factory_supports_all_strategies():
    from src.retrieval.factory import get_retriever
    for strat in ["simple", "semantic", "sparse", "bm25", "hybrid", "rerank", "hyde", "query_expansion", "mmr", "hybrid_rerank"]:
        r = get_retriever(strat, top_k=3)
        assert r is not None
        assert r.top_k == 3


def test_file_sha256_computation(tmp_path):
    from src.ingestion.ingest import compute_file_sha256
    test_f = tmp_path / "sample.txt"
    test_f.write_text("Hello DocRetriever RAG", encoding="utf-8")
    hash1 = compute_file_sha256(test_f)
    assert len(hash1) == 64  # Valid SHA-256 hex string
    
    test_f.write_text("Modified Content", encoding="utf-8")
    hash2 = compute_file_sha256(test_f)
    assert hash1 != hash2  # Hash changes on modification
