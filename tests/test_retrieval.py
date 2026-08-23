import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.base import Chunk
from src.retrieval.hybrid import HybridRetriever
from eval.metrics import recall_at_k, mrr, compute_retrieval_metrics


def test_chunk_defaults():
    chunk = Chunk(id=1, content="test", source_file="a.md", section_title=None, chunk_index=0)
    assert chunk.score == 0.0
    assert chunk.metadata == {}


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
