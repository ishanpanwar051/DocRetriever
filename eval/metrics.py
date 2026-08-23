"""
eval/metrics.py — Custom Retrieval Metrics for DocRetriever

WHY these metrics:
- Recall@k: Proportion of queries where the expected document is in top-k.
- MRR (Mean Reciprocal Rank): 1/rank of the first relevant document.
  Measures ranking quality (placing the true document at rank 1 is rewarded much more than rank 5).
"""


def recall_at_k(retrieved_sources: list[str], expected_source: str, k: int = 5) -> float:
    """1.0 if expected_source is among top-k retrieved sources, else 0.0."""
    if not expected_source:
        return 0.0
    top_k = retrieved_sources[:k]
    # Check exact match or basename/relative match
    exp_clean = expected_source.replace("\\", "/").strip().lower()
    for s in top_k:
        s_clean = s.replace("\\", "/").strip().lower()
        if exp_clean in s_clean or s_clean in exp_clean:
            return 1.0
    return 0.0


def mrr(retrieved_sources: list[str], expected_source: str) -> float:
    """Computes reciprocal rank: 1 / (rank of first match)."""
    if not expected_source:
        return 0.0
    exp_clean = expected_source.replace("\\", "/").strip().lower()
    for rank, s in enumerate(retrieved_sources, 1):
        s_clean = s.replace("\\", "/").strip().lower()
        if exp_clean in s_clean or s_clean in exp_clean:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(eval_results: list[dict]) -> dict:
    """
    Computes aggregated Recall@5, Recall@3, and MRR over answerable & edge test cases.
    """
    eval_cases = [r for r in eval_results if r.get("type") in ("answerable", "edge") and r.get("expected_source")]
    if not eval_cases:
        return {"recall_at_5": 0.0, "recall_at_3": 0.0, "mrr": 0.0, "num_evaluated": 0}

    r5_total = sum(recall_at_k(r.get("retrieved_sources", []), r["expected_source"], k=5) for r in eval_cases)
    r3_total = sum(recall_at_k(r.get("retrieved_sources", []), r["expected_source"], k=3) for r in eval_cases)
    mrr_total = sum(mrr(r.get("retrieved_sources", []), r["expected_source"]) for r in eval_cases)

    n = len(eval_cases)
    return {
        "recall_at_5": round(r5_total / n, 4),
        "recall_at_3": round(r3_total / n, 4),
        "mrr": round(mrr_total / n, 4),
        "num_evaluated": n,
    }
