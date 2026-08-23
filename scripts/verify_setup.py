"""
scripts/verify_setup.py — Phase A verification script

Checks ALL Phase A components:
  ✅ Config loads
  ✅ PostgreSQL reachable
  ✅ pgvector extension enabled
  ✅ Tables exist
  ✅ Ollama server running
  ✅ nomic-embed-text (embedding + dimension test)
  ✅ llama3.2:3b available
  ✅ RAM management utility
  ✅ sentence-transformers CrossEncoder importable (reranker)
  ✅ Corpus downloaded

USAGE:
  python scripts/verify_setup.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

CHECKS = []


def check(name: str, critical: bool = True):
    def decorator(fn):
        CHECKS.append({"name": name, "fn": fn, "critical": critical})
        return fn
    return decorator


# ── 1: Config ──────────────────────────────────────────────────────────────────
@check("Config loads (settings.py)")
def check_config():
    from config.settings import settings
    assert settings.postgres_host
    assert settings.ollama_base_url
    assert settings.ollama_embed_model == "nomic-embed-text"
    assert settings.ollama_keep_alive == 0, "keep_alive must be 0 for RAM management!"
    return f"DB={settings.postgres_host}:{settings.postgres_port} | LLM={settings.ollama_llm_model} | keep_alive={settings.ollama_keep_alive}"


# ── 2: PostgreSQL ──────────────────────────────────────────────────────────────
@check("PostgreSQL connection")
def check_postgres():
    import psycopg2
    from config.settings import settings
    conn = psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db, connect_timeout=5,
    )
    cur = conn.cursor()
    cur.execute("SELECT version()")
    ver = cur.fetchone()[0]
    conn.close()
    return f"Connected ✓ | {ver[:50]}"


# ── 3: pgvector ────────────────────────────────────────────────────────────────
@check("pgvector extension enabled")
def check_pgvector():
    import psycopg2
    from config.settings import settings
    conn = psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    cur = conn.cursor()
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    conn.close()
    assert row, "pgvector NOT installed! Run: docker compose down -v && docker compose up -d"
    return f"pgvector v{row[0]} ✓"


# ── 4: Tables ──────────────────────────────────────────────────────────────────
@check("Database tables created")
def check_tables():
    import psycopg2
    from config.settings import settings
    conn = psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename IN ('document_chunks', 'eval_runs')
    """)
    tables = {r[0] for r in cur.fetchall()}

    # Also check HNSW index exists
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'document_chunks'
        AND indexname = 'idx_chunks_embedding_hnsw'
    """)
    hnsw = cur.fetchone()
    conn.close()

    missing = {"document_chunks", "eval_runs"} - tables
    assert not missing, f"Missing tables: {missing}. Did init_db.sql run on first docker compose up?"
    hnsw_status = "HNSW index ✓" if hnsw else "⚠️ HNSW index missing"
    return f"Tables: {sorted(tables)} | {hnsw_status}"


# ── 5: Ollama server ───────────────────────────────────────────────────────────
@check("Ollama server running")
def check_ollama_server():
    import httpx
    from config.settings import settings
    resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    models = [m["name"] for m in resp.json().get("models", [])]
    return f"Running ✓ | Pulled models: {models or ['(none yet — run setup_ollama.ps1)']}"


# ── 6: nomic-embed-text ────────────────────────────────────────────────────────
@check("nomic-embed-text — embedding dimension test")
def check_embed_model():
    import ollama
    from config.settings import settings
    resp = ollama.embed(
        model=settings.ollama_embed_model,
        input="DocRetriever Phase A setup verification test",
    )
    # Handle both ollama SDK versions
    if hasattr(resp, "embeddings"):
        embeddings = resp.embeddings
    else:
        embeddings = resp.get("embeddings", [])

    assert embeddings, "No embeddings returned from Ollama"
    dim = len(embeddings[0])
    assert dim == 768, f"Expected 768-dim, got {dim}. Wrong model? Check OLLAMA_EMBED_MODEL in .env"
    return f"768-dim embedding ✓ | model={settings.ollama_embed_model}"


# ── 7: llama3.2:3b ────────────────────────────────────────────────────────────
@check("llama3.2:3b model available")
def check_llm():
    import httpx
    from config.settings import settings
    resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    llm = settings.ollama_llm_model
    found = any(llm.replace(":latest", "") in m for m in models)
    if not found:
        raise AssertionError(
            f"{llm} not found. Run: ollama pull {llm}\n"
            f"Or run: .\\scripts\\setup_ollama.ps1\n"
            f"Available: {models}"
        )
    return f"{llm} found ✓"


# ── 8: RAM utility ────────────────────────────────────────────────────────────
@check("RAM management utility (memory.py)")
def check_memory():
    from src.utils.memory import get_ram_usage_gb, OllamaModelManager
    from config.settings import settings

    ram = get_ram_usage_gb()
    mgr = OllamaModelManager(base_url=settings.ollama_base_url)
    loaded = mgr.list_loaded()

    if "error" in ram:
        return f"psutil unavailable — install: pip install psutil | Loaded: {loaded}"

    warn = " ⚠️ LOW — close other apps!" if ram["available_gb"] < 3.5 else " ✓"
    return (
        f"RAM: {ram['used_gb']}/{ram['total_gb']} GB "
        f"({ram['available_gb']} GB free{warn}) | "
        f"Ollama loaded: {loaded or 'none'}"
    )


# ── 9: CrossEncoder (reranker) importable ─────────────────────────────────────
@check("sentence-transformers CrossEncoder importable (reranker)")
def check_reranker_import():
    # WHY sentence-transformers not FlagEmbedding:
    # FlagEmbedding has install conflicts on some Py3.11 setups.
    # CrossEncoder from sentence-transformers loads bge-reranker-base identically.
    from sentence_transformers import CrossEncoder  # noqa
    # Don't actually load the model here — it downloads ~0.3GB on first use
    return "CrossEncoder importable ✓ (model downloads on first rerank call)"


# ── 10: Corpus ────────────────────────────────────────────────────────────────
@check("FastAPI corpus downloaded", critical=False)
def check_corpus():
    corpus_dir = Path("corpus/fastapi_docs")
    if not corpus_dir.exists():
        raise AssertionError(
            "Corpus dir missing. Run: python scripts/download_corpus.py"
        )
    md_files = list(corpus_dir.rglob("*.md"))
    if len(md_files) < 10:
        raise AssertionError(
            f"Only {len(md_files)} .md files (expected 80+). "
            "Run: python scripts/download_corpus.py"
        )
    total_kb = sum(f.stat().st_size for f in md_files) / 1024
    return f"{len(md_files)} .md files | {total_kb:.1f} KB | path={corpus_dir.absolute()}"


# ── Runner ────────────────────────────────────────────────────────────────────
def run_all_checks():
    print("\n" + "=" * 65)
    print("  DocRetriever — Phase A Verification")
    print("  Run from project root with venv activated")
    print("=" * 65)

    results = []
    critical_failed = False

    for c in CHECKS:
        name, fn, critical = c["name"], c["fn"], c["critical"]
        try:
            detail = fn()
            status = "✅ PASS"
        except AssertionError as e:
            status = "❌ FAIL" if critical else "⚠️  SKIP"
            detail = str(e)
            if critical:
                critical_failed = True
        except ImportError as e:
            status = "❌ ERR " if critical else "⚠️  WARN"
            detail = f"Missing package: {e}"
            if critical:
                critical_failed = True
        except Exception as e:
            status = "❌ ERR " if critical else "⚠️  WARN"
            detail = f"{type(e).__name__}: {e}"
            if critical:
                critical_failed = True

        results.append((status, name, detail or ""))

    print()
    for status, name, detail in results:
        print(f"  {status}  {name}")
        if detail:
            lines = detail.split("\n")
            for line in lines[:3]:  # max 3 lines per check
                short = line[:75] + "…" if len(line) > 75 else line
                print(f"            {short}")

    print()
    print("=" * 65)
    if not critical_failed:
        print("  🎉 ALL CRITICAL CHECKS PASSED — Phase A complete!")
        print("     Next: Phase B → ingestion + baseline eval (~60%)")
    else:
        print("  ❌ Critical checks failed. Fix issues above, then re-run.")
        print("     Tip: docker compose logs postgres  (for DB issues)")
        print("     Tip: ollama serve  (if Ollama not responding)")
    print("=" * 65 + "\n")
    return not critical_failed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
