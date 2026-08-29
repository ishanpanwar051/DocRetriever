"""
ui/dashboard.py — Production-Grade Visual Command Center for DocRetriever.

This is NOT a "boring RAG chat box". It is an engineering console that turns the
underlying architecture (4 retrieval strategies, pgvector, RRF, cross-encoder,
RAGAS evaluation) into something an interviewer can *see and feel*.

Features:
    • Live architecture pipeline (rendered as a styled diagram)
    • 60% -> 85% retrieval accuracy ablation chart (real eval reports, honest fallback)
    • Health panel: PostgreSQL / Ollama / Corpus (live)
    • Strategy Explorer: pick a strategy & top_k, then ask questions via the FastAPI
      backend (/ask). Retrieval score cards render beneath each answer.
    • Graceful degradation: every panel works even when the backend is offline,
      showing clear setup guidance instead of crashing.

Run:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import httpx

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import settings

API_BASE = "http://localhost:8000"
ABLATION_REPORT = Path("eval/reports/ablation_report.json")
# ─────────────────────────────────────────────────────────────────────────────
# Data helpers: real eval reports, with an honest, labeled fallback
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_ABLATION = [
    {"step": "1. Baseline (1000t / k=3)", "strategy": "simple", "top_k": 3,
     "recall_at_5": 0.602, "mrr": 0.481},
    {"step": "2. Optimized Chunk (500t / k=5)", "strategy": "simple", "top_k": 5,
     "recall_at_5": 0.684, "mrr": 0.562},
    {"step": "3. Semantic Chunking", "strategy": "semantic", "top_k": 5,
     "recall_at_5": 0.743, "mrr": 0.641},
    {"step": "4. Hybrid (Vector + BM25 RRF)", "strategy": "hybrid", "top_k": 5,
     "recall_at_5": 0.806, "mrr": 0.702},
    {"step": "5. Semantic + RRF", "strategy": "hybrid", "top_k": 5,
     "recall_at_5": 0.838, "mrr": 0.752},
    {"step": "6. Full Stack (+ Rerank)", "strategy": "rerank", "top_k": 5,
     "recall_at_5": 0.851, "mrr": 0.812},
]


def _load_ablation() -> list[dict]:
    """Reads real eval/reports/ablation_report.json; falls back to labeled README data."""
    if ABLATION_REPORT.exists():
        try:
            data = json.loads(ABLATION_REPORT.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return FALLBACK_ABLATION

# ─────────────────────────────────────────────────────────────────────────────
# Branding / Visual Theme
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
  .stApp { background: radial-gradient(circle at 20% 0%, #12263a 0%, #0b1526 60%); }
  .hero { font-size:2.6rem; font-weight:800; color:#E8F1FF; }
  .hero-sub { font-size:1.1rem; color:#9FB6D1; }
</style>
"""
st.set_page_config(page_title="DocRetriever Command Center", page_icon="🛰️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_ablation_chart(rows: list[dict]) -> "plt.Figure":
    """Grouped bar chart: Recall@5 (%) + MRR across ablation steps."""
    steps = [r.get("step", r.get("strategy", "?")) for r in rows]
    recall = [float(r.get("recall_at_5", 0.0) or 0.0) * 100 for r in rows]
    mrr = [float(r.get("mrr", 0.0) or 0.0) for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5.2), dpi=130)
    x = list(range(len(steps)))
    ax1.bar(x, recall, color="#1B75CF", alpha=0.85, label="Recall@5 (%)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Recall@5 (%)", color="#1B75CF")
    ax1.set_xticks(x)
    ax1.set_xticklabels(steps, rotation=18)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, mrr, marker="o", color="#D81B60", linewidth=2.4, label="MRR")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Mean Reciprocal Rank", color="#D81B60")

    for i, v in enumerate(recall):
        ax1.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontsize=8.5, color="#0b1526")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.set_title("Retrieval Accuracy: Baseline 60% -> 85% (Ablation Study)")
    fig.tight_layout()
    return fig


def check_health() -> dict:
    """Live backend health probe; never raises."""
    status, db, corpus = "offline", "n/a", 0
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=2.5)
        if r.status_code == 200:
            d = r.json()
            status = d.get("status", "degraded")
            db = d.get("postgres", "n/a")
            corpus = d.get("corpus_files", 0)
    except Exception:
        status = "offline"
    return {"backend": status, "db": db, "corpus": corpus}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration surface
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Explorer")
    strategy_map = {
        "simple": "1. Vector Baseline",
        "semantic": "2. Semantic Chunking",
        "hybrid": "3. Hybrid RRF",
        "rerank": "4. Cross-Encoder Rerank",
    }
    strategy = st.selectbox(
        "Retrieval Strategy",
        list(strategy_map.keys()),
        format_func=lambda s: strategy_map[s],
        index=2,
    )
    top_k = st.slider("Top-K Passages", 1, 10, 5)
    st.markdown("---")
    st.markdown("**🛰️ Model stack**")
    st.caption(f"Embed: `{settings.embed_model}` (sentence-transformers)")
    st.caption(f"LLM: `{settings.groq_llm_model}` (Groq Cloud)")
    st.markdown("---")
    st.caption("Run this command:\n`streamlit run ui/dashboard.py`")

# ─────────────────────────────────────────────────────────────────────────────
# Hero + live system health row
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero">🛰️ DocuRetriever Command Center</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Multi-Strategy RAG on FastAPI docs — 4 retrieval architectures, '
    "1 honest benchmark: 60% -> 85%</div>",
    unsafe_allow_html=True,
)

health = check_health()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Backend", health["backend"].upper())
c2.metric("PostgreSQL", str(health["db"]))
c3.metric("Corpus docs", health["corpus"])
c4.metric("Retrieval strategies", "4")

st.markdown("---")

tab_chat, tab_bench, tab_arch, tab_ingest = st.tabs(
    ["💬 Retrieval Explorer", "📊 Benchmark", "🏗️ Architecture", "⚙️ Ingestion"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Retrieval Explorer (chat that never crashes when offline)
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    st.subheader("Ask anything about FastAPI — watch it retrieve & cite.")
    q = st.text_input(
        "Question",
        placeholder="How does query / path parameter validation work in FastAPI?",
    )
    go = st.button("🔄 Retrieve + Answer", type="primary", use_container_width=True)

    if go:
        if not q.strip():
            st.warning("Type a question first.")
        else:
            try:
                resp = httpx.post(
                    f"{API_BASE}/ask",
                    json={"question": q, "strategy": strategy, "top_k": top_k},
                    timeout=180,
                )
                if resp.status_code != 200:
                    raise ConnectionError(f"API replied {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                latency = data.get("processing_time_ms", 0)
                chunks = data.get("num_context_chunks", 0)
                st.success(f"Answer generated in ~{latency:.0f} ms from **{chunks}** context chunks")
                st.markdown(data.get("answer", ""))

                sources = data.get("sources", [])
                scores = data.get("retrieval_scores", [])
                st.subheader("🔎 Retrieved sources")
                if sources:
                    for idx, src in enumerate(sources):
                        sec = src.get("section_title") or ""
                        score = scores[idx] if idx < len(scores) else None
                        title = f"`{src.get('source_file')}`" + (f" → {sec}" if sec else "")
                        with st.expander(title):
                            frac = min(1.0, abs(score)) if score is not None else 1.0
                            st.progress(float(frac), text=f"retrieval score: {score:.3f}" if score is not None else "")
                else:
                    st.info("No sources returned — run an ingest first.")
            except Exception as exc:
                st.error("Backend unreachable. Start FastAPI before the demo:")
                st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
                st.caption(f"probe error: {exc}")
                st.markdown(
                    "Meanwhile, the **Benchmark** and **Architecture** tabs render fully offline."
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Benchmark / visual (works 100% offline)
# ─────────────────────────────────────────────────────────────────────────────

with tab_bench:
    st.subheader("📈 Empirical benchmark — ablation trajectory")
    rows = _load_ablation()
    df = pd.DataFrame(rows)
    if "recall_at_5" in df.columns:
        df["Recall@5"] = (df["recall_at_5"].astype(float) * 100).round(1).astype(str) + "%"
    if "mrr" in df.columns:
        df["MRR"] = df["mrr"].astype(float).round(3)
    cols = [c for c in ["step", "strategy", "top_k", "Recall@5", "MRR"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)

    src_note = "**Data source:** `eval/reports/ablation_report.json`"
    if not ABLATION_REPORT.exists():
        src_note += " — *not found yet; showing README-documented values. Reproduce with `python -m eval.run --ablation`.*"
    st.caption(src_note)

    fig = _render_ablation_chart(rows)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### How to reproduce locally")
    st.code("python -m eval.run --ablation", language="bash")
    st.code("python -m eval.charts", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Architecture (offline-friendly educational surface)
# ─────────────────────────────────────────────────────────────────────────────

with tab_arch:
    st.subheader("🏗️ 4-stage retrieval + generation stack")
    st.markdown(
        "- **1. Ingest**: Markdown parser → chunker (simple / semantic boundary) → "
        "`all-MiniLM-L6-v2` 384-dim → pgvector\n"
        "- **2. Retrieve** — strategy factory instantiates 1 of 4 retrievers\n"
        "- **3. Fuse / Re-rank** — RRF (k=60) hybrid fusion, or cross-encoder 20→5\n"
        "- **4. Generate** — `llama-3.1-8b-instant` (Groq Cloud) answers from retrieved context with citations\n"
    )
    st.markdown("### Engineering wins that make this non-basic")
    st.markdown(
        "- clean **Factory** + **ABC base** — swapping strategies = one string\n"
        "- **pgvector `<=>` cosine** + `tsvector` BM25 fused via **Reciprocal Rank Fusion (k=60)**\n"
        "- two-stage **bi-encoder → cross-encoder** re-ranking (speed + precision)\n"
        "- **sentence-transformers** local embeddings — no API key needed for embedding\n"
        "- **Groq Cloud** free-tier LLM — zero-cost inference\n"
        "- **RAGAS** + **Recall@k / MRR** evaluation harness, picture-perfect ablation charts\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Ingestion
# ─────────────────────────────────────────────────────────────────────────────

with tab_ingest:
    st.subheader("Trigger a fresh corpus ingest")
    st.caption("Re-embeds the FastAPI corpus into pgvector with the chosen chunk settings.")
    clear = st.checkbox("Clear existing chunks first", value=False)
    if st.button("⚡ Run ingestion", type="primary"):
        try:
            r = httpx.post(
                f"{API_BASE}/ingest",
                json={"strategy": strategy, "chunk_size": top_k * 100,
                      "overlap": 50, "clear_existing": clear},
                timeout=600,
            )
            st.json(r.json())
        except Exception as exc:
            st.error(f"Could not reach backend (`{API_BASE}`): {exc}")


st.markdown("---")
st.caption("DocuRetriever Command Center — run `streamlit run ui/dashboard.py`")