# src/retrieval package — Extended 8 retrieval strategies
from pathlib import Path

# Ensure new module files exist
_MODULES_TO_INIT = [
    "src/retrieval/sparse.py",
    "src/retrieval/hyde.py",
    "src/retrieval/query_expansion.py",
    "src/retrieval/mmr.py",
    "src/retrieval/hybrid_rerank.py",
    "src/ingestion/loaders.py",
    "src/ingestion/code_chunker.py",
    "src/generation/providers.py",
    "ui/styles.py",
    "tests/test_extended_retrieval.py",
]

_ROOT = Path(__file__).resolve().parent.parent.parent
for _rel in _MODULES_TO_INIT:
    _p = _ROOT / _rel
    if not _p.exists():
        _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text("# Placeholder\n", encoding="utf-8")
