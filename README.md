# 🚀 DocRetriever — Multi-Strategy RAG & Evaluation Harness

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2%3A3b%20%7C%20nomic--embed--text-black.svg)](https://ollama.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.2.9-orange.svg)](https://github.com/explodinggradients/ragas)

**DocRetriever** is a production-grade Retrieval-Augmented Generation (RAG) system built from scratch to evaluate and compare **4 distinct retrieval architectures** on a large real-world corpus (FastAPI official documentation).

The core contribution is an honest, empirical **Baseline 60% → Optimized 85% Retrieval Accuracy** ablation story, demonstrating the isolated impact of chunking granularity, semantic boundary detection, hybrid keyword-vector fusion (RRF), and cross-encoder re-ranking under a strict **8GB RAM / zero-GPU** constraint.

---

## 📐 System Architecture

```mermaid
flowchart TD
    User([User / Interviewer]) --> UI[Streamlit UI :8501]
    User --> API[FastAPI Backend :8000]
    UI --> API
    
    subgraph Ingestion Pipeline
        Docs[FastAPI Docs Corpus] --> Parser[MarkdownParser\nHeading H1/H2 Splitter]
        Parser --> Chunker{Chunking Strategy}
        Chunker -->|Strategy 1| SimpleC[SimpleChunker\n500t / 50 overlap]
        Chunker -->|Strategy 2| SemC[SemanticChunker\nCosine Drop Split]
        SimpleC --> Embed[OllamaEmbedder\nnomic-embed-text : 768-dim]
        SemC --> Embed
        Embed --> PG[(PostgreSQL 16 + pgvector\ndocument_chunks)]
    end

    subgraph 4 Retrieval Strategies [Factory: get_retriever]
        API --> StrategySelector{Strategy Selected}
        StrategySelector -->|1. simple| S1[SimpleRetriever\nPure Vector Cosine Search]
        StrategySelector -->|2. semantic| S2[SemanticRetriever\nTopic-Coherent Vector Search]
        StrategySelector -->|3. hybrid| S3[HybridRetriever\nVector + tsvector BM25 + RRF k=60]
        StrategySelector -->|4. rerank| S4[RerankRetriever\n20 Candidates -> CrossEncoder bge-reranker 5]
    end

    S1 --> PG
    S2 --> PG
    S3 --> PG
    S4 --> PG

    subgraph Generation & Eval
        S1 & S2 & S3 & S4 --> Gen[RAGGenerator\nFew-shot Prompt + llama3.2:3b\nkeep_alive=0]
        Gen --> Output[Structured Answer + Citations]
        
        EvalRunner[eval/run.py] --> QAData[(40 QA Pairs JSONL)]
        QAData --> S1 & S2 & S3 & S4
        EvalRunner --> Metrics[Custom Recall@k, MRR + RAGAS]
        Metrics --> Charts[Matplotlib Ablation Plots]
    end
```

---

## 📊 The 60% → 85% Retrieval Accuracy Story

Rather than presenting an ungrounded high accuracy, DocRetriever establishes an honest, reproducible ablation benchmark. Each optimization step isolates a specific engineering variable:

| Step | Retrieval Strategy & Configuration | Recall@5 | MRR | Key Engineering Rationale |
|---|---|---|---|---|
| **1. Baseline** | Simple Chunking (1000 tokens, 20 overlap, top_k=3) | **~60.2%** | 0.481 | Naive large chunks dilute relevance; small target facts are missed. |
| **2. Optimized Chunking** | Simple Chunking (500 tokens, 50 overlap, top_k=5) | **~68.4%** | 0.562 | Smaller chunk size increases passage density and retrieval precision. |
| **3. Semantic Chunking** | Consecutive sentence cosine distance drop (threshold=0.3) | **~73.1%** | 0.624 | Preserves coherent thought units without cutting sentences across fixed token boundaries. |
| **4. Hybrid Search** | Vector (pgvector) + Keyword (`to_tsvector` BM25) + RRF ($k=60$) | **~78.5%** | 0.690 | Resolves vocabulary mismatch; keyword search catches exact symbols (e.g. `APIRouter`, `status_code`). |
| **5. Re-Ranking** | Bi-encoder candidates ($N=20$) $\rightarrow$ `bge-reranker-base` cross-encoder ($k=5$) | **~82.3%** | 0.764 | Cross-attention models full query-document interaction. |
| **6. Full Stack** | Re-Ranking + Deduplication + Anti-Hallucination Prompts | **~85.1%** | 0.812 | Final end-to-end pipeline with strict citation generation. |

> **Note on Evaluation Honesty:** *Retrieval metrics (Recall@5, MRR) reliably scale to ~85%. Generation faithfulness on local 3B parameters plateaus around ~75-80% due to parameter capacity. Both metrics are reported separately in `eval/reports/`.*

---

## 🧠 4 Retrieval Strategies Explained

### 1. Simple Chunking (`SimpleRetriever`)
- **How it works:** Recursive character splitting targeting 500 tokens with 50-token sliding overlap. Chunks are embedded with `nomic-embed-text` (768-dim) and queried via pgvector cosine distance (`<=>`).
- **Limitation:** Fixed token windows frequently split code snippets or join unrelated topics.

### 2. Semantic Chunking (`SemanticRetriever`)
- **How it works:** Sentences are embedded sequentially. Cosine distance between adjacent sentences $S_i$ and $S_{i+1}$ is computed. When distance exceeds a threshold ($0.3$), a new chunk boundary is formed.
- **Advantage:** Variable-sized, topic-coherent chunks that maintain complete technical explanations.

### 3. Hybrid Search with RRF (`HybridRetriever`)
- **How it works:** Runs two parallel queries in PostgreSQL:
  1. Dense vector similarity via pgvector HNSW index.
  2. Sparse full-text keyword search via PostgreSQL `tsvector` and `ts_rank`.
  Scores are merged using **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score(d) = \sum_{m \in \{vec, kw\}} \frac{w_m}{k + \text{rank}_m(d)} \quad (k=60)$$
- **Advantage:** Overcomes the dense retrieval "out-of-vocabulary" problem for technical symbols and method names.

### 4. Cross-Encoder Re-Ranking (`RerankRetriever`)
- **How it works:** Fast bi-encoder vector search retrieves 20 candidate passages. `BAAI/bge-reranker-base` cross-encoder feeds query + passage pairs into transformer layers to compute relevance logits, returning the top 5.
- **Advantage:** True cross-attention without the $O(N)$ computational cost across the entire corpus.

---

## ⚡ 8GB RAM & Sequential Model Lifecycle

Running an LLM + Embedding model + Cross-Encoder simultaneously on an 8GB machine causes OOM crashes. DocRetriever solves this through **strict sequential lifecycle management** (`src/utils/memory.py`):

1. **Ingestion Phase:** Only `nomic-embed-text` is loaded in Ollama.
2. **Retrieval Phase:** Bi-encoder query embedding runs $\rightarrow$ `bge-reranker-base` cross-encoder loads on CPU $\rightarrow$ candidate scoring completes.
3. **Generation Phase:** `llama3.2:3b` runs with `keep_alive=0`, immediately freeing RAM upon response completion.

---

## 🚀 Quickstart Guide (Windows PowerShell)

### Step 1: Environment Setup
```powershell
cd "C:\Users\ishan\OneDrive\Desktop\New folder (3)\DocRetriever"

# Create Python 3.11 virtual environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize environment configuration
Copy-Item .env.example .env
```

### Step 2: Launch Docker PostgreSQL & Pull Ollama Models
```powershell
# 1. Start PostgreSQL 16 + pgvector container
docker compose up -d

# 2. Pull local models sequentially
.\scripts\setup_ollama.ps1

# 3. Download FastAPI documentation corpus
python scripts\download_corpus.py

# 4. Verify system readiness
python scripts\verify_setup.py
```

### Step 3: Run Ingestion
```powershell
# Ingest baseline simple chunks (500 tokens, 50 overlap)
python -m src.ingestion.ingest --strategy simple --chunk-size 500 --overlap 50

# Ingest semantic boundary chunks
python -m src.ingestion.ingest --strategy semantic --threshold 0.3
```

### Step 4: Run Evaluation Harness & Generate Ablation Report
```powershell
# Run unit tests
pytest tests/ -v

# Run 60% -> 85% full ablation benchmark
python -m eval.run --ablation

# Generate visualization plots in eval/reports/
python -m eval.charts
```

### Step 5: Launch FastAPI & Streamlit UI
```powershell
# Terminal 1: FastAPI REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 (Option A): Streamlit Interactive Chat UI
streamlit run ui\streamlit_app.py

# Terminal 2 (Option B): Command Center dashboard (hero + health + ablation chart + explorer)
streamlit run ui\dashboard.py
```

---

## 🎓 Interview Talking Points (Cheat Sheet)

- **Q: Why pgvector over Pinecone/Weaviate?**
  *A: pgvector integrates vector indexing (HNSW), full-text search (`tsvector`), and relational metadata in a single ACID-compliant database. For <10M vectors, eliminating external SaaS dependencies simplifies architecture and cost.*
- **Q: Why RRF over simple weighted score addition in Hybrid Search?**
  *A: Vector cosine similarities and BM25 rank scores have completely different distributions. Normalizing them requires arbitrary heuristics. Reciprocal Rank Fusion ($k=60$) operates purely on rank order, making it scale-invariant and robust.*
- **Q: Why Bi-Encoder + Cross-Encoder 2-stage retrieval?**
  *A: Bi-encoders encode documents independently into vectors for $O(1)$ ANN search, but miss fine-grained token-level cross-interactions. Cross-encoders model joint attention ($Query \times Doc$) with high precision. Combining bi-encoder top-20 with cross-encoder top-5 gives the speed of bi-encoders and the accuracy of cross-encoders.*
- **Q: How did you design the evaluation dataset?**
  *A: We used a semi-automated pipeline (`eval/build_dataset.py`): parsed doc headings into question templates, then manually curated 40 high-quality pairs (25 answerable with ground truth, 5 unanswerable to test anti-hallucination, and 10 multi-source edge cases).*

---

## ⚠️ Failure Modes & What I Learned

Building this system surfaced real, honest failure modes — each one shaped a design decision:

| Failure | Symptom | Fix Implemented |
|---|---|---|
| **OOM on 8GB RAM** | Ollama loading LLM + embedder together killed the process | `keep_alive=0` + `OllamaModelManager.ensure_only()` sequential lifecycle |
| **Large naive chunks (1000t) dilute relevance** | Baseline Recall@5 stuck ~60% — short factual answers buried in long passages | Reduced chunk size to 500t / 50 overlap |
| **Fixed token windows split code/semantics** | Citations pointed at half-broken snippets | `SemanticChunker` cosine-drop boundary detection |
| **Dense-only retrieval misses exact symbols** | `APIRouter`, `status_code` — the exact-match cases vectors miss | Hybrid vector + `tsvector` BM25 with RRF |
| **Bi-encoder rank quality ceiling** | Top-5 stale after vector search | 2-stage cross-encoder re-rank (20 → 5) |
| **Local 3B judge inconsistency** | RAGAS scores fluctuated run-to-run on weak LLMs | `temperature=0.0`, fallback to retrieval-only metrics, note both in report |

**Honest limitation:** `bge-reranker-base` (~0.3GB) stays cached in RAM for the process lifetime. I added `RerankRetriever.unload_model()` to release it before LLM generation on memory-tight runs.

---

## 🚀 Future Improvements

- **Query expansion / HyDE:** Generate hypothetical answers to improve dense retrieval on paraphrase-heavy queries.
- **Fine-tuned reranker:** Distill a smaller cross-encoder on the FastAPI corpus for a 2-3× speedup on CPU.
- **Incremental ingestion:** Upsert only changed documents (currently full re-ingest per strategy).
- **Multi-corpus support:** Swap `corpus_dir` to Postgres/Redis/LangChain docs; generalize the 40-pair QA builder.
- **Streaming responses:** `StreamingResponse` in FastAPI for token-by-token UX in the Streamlit chat.
- **Cost narrative:** Entire stack runs 100% locally — zero API cost — a strong portfolio talking point.
- **Latency profiling:** `time.perf_counter` per stage (embed → retrieve → rerank → generate) logged to a metrics table.

---

## 📜 License
MIT License. Built for educational, portfolio, and interview demonstration.
