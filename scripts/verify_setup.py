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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    assert settings.embed_model
    assert settings.embedding_dim == 384
    return f"DB={settings.postgres_host}:{settings.postgres_port} | Embed={settings.embed_model} (dim={settings.embedding_dim}) | LLM={settings.groq_llm_model}"


# ── 2: PostgreSQL ──────────────────────────────────────────────────────────────
@check("PostgreSQL connection", critical=False)
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
@check("pgvector extension enabled", critical=False)
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
@check("Database tables created", critical=False)
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


# ── 5: Local Embedder (sentence-transformers) ──────────────────────────────────
@check("Local Embedder (all-MiniLM-L6-v2 — 384-dim test)")
def check_embed_model():
    from src.ingestion.embedder import LocalSentenceEmbedder
    from config.settings import settings
    embedder = LocalSentenceEmbedder(model=settings.embed_model)
    emb = embedder.embed_single("DocRetriever verification test")
    dim = len(emb)
    assert dim == settings.embedding_dim, f"Expected {settings.embedding_dim}-dim, got {dim}."
    return f"{dim}-dim embedding verified ✓ | model={settings.embed_model}"


# ── 6: LLM Provider (Groq / Ollama) ───────────────────────────────────────────
@check("LLM Provider availability", critical=False)
def check_llm():
    from config.settings import settings
    import httpx
    if settings.groq_api_key:
        resp = httpx.get(
            f"{settings.groq_base_url}/models",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=5,
        )
        if resp.status_code == 200:
            return f"Groq Cloud API connected ✓ | Model: {settings.groq_llm_model}"
        return f"Groq API returned status {resp.status_code}"
    
    # Fallback check for local Ollama
    resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
    models = [m["name"] for m in resp.json().get("models", [])]
    return f"Local Ollama running ✓ | Models: {models}"


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
