"""
api/main.py — FastAPI Backend for DocRetriever

Exposes endpoints:
- GET  /health          → Health check for DB, Groq API, and Corpus
- POST /ingest          → Trigger corpus ingestion
- POST /ask             → Query DocRetriever using any of 4 strategies
- GET  /eval/results    → Retrieve latest eval benchmarks
"""

import time
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.schemas import (
    AskRequest, AskResponse, IngestRequest, IngestResponse,
    HealthResponse, EvalResultsResponse, SourceCitation
)
from src.retrieval.factory import get_retriever
from src.generation.generator import RAGGenerator
from src.ingestion.ingest import ingest_corpus
from src.db.connection import test_connection

app = FastAPI(
    title="DocRetriever API",
    description="Production-grade RAG System with 4 Evaluated Retrieval Strategies.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Checks PostgreSQL, Groq API, and Corpus file count."""
    try:
        db_ok = test_connection()
        pg_status = "connected" if db_ok else "disconnected"
    except Exception as e:
        pg_status = f"error: {e}"

    import httpx
    try:
        resp = httpx.get(
            f"{settings.groq_base_url}/models",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=5,
        )
        groq_status = "connected" if resp.status_code == 200 else "unreachable"
    except Exception:
        groq_status = "unreachable"

    corpus_dir = Path(settings.corpus_dir)
    corpus_files = len(list(corpus_dir.rglob("*.md"))) if corpus_dir.exists() else 0

    return HealthResponse(
        status="ok" if pg_status == "connected" and groq_status == "connected" else "degraded",
        postgres=pg_status,
        groq_api=groq_status,
        corpus_files=corpus_files,
        timestamp=datetime.now(),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Triggers corpus ingestion with chosen strategy and parameters."""
    start_time = time.time()
    try:
        files_count, chunks_count = ingest_corpus(
            corpus_dir=settings.corpus_dir,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            clear_existing=request.clear_existing,
        )
        duration = time.time() - start_time
        return IngestResponse(
            status="success",
            files_processed=files_count,
            chunks_created=chunks_count,
            strategy=request.strategy,
            processing_time_seconds=round(duration, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Query the system using a specific strategy."""
    start_time = time.time()
    try:
        # 1. Instantiate retriever for chosen strategy
        retriever = get_retriever(strategy=request.strategy, top_k=request.top_k)

        # 2. Retrieve relevant chunks
        chunks = retriever.retrieve(request.question)

        # 3. Generate answer with Groq API
        generator = RAGGenerator(model=settings.groq_llm_model)
        rag_resp = generator.generate(
            question=request.question,
            chunks=chunks,
            strategy=request.strategy,
        )

        duration_ms = (time.time() - start_time) * 1000

        return AskResponse(
            answer=rag_resp.answer,
            sources=[SourceCitation(source_file=s.source_file, section_title=s.section_title) for s in rag_resp.sources],
            strategy=request.strategy,
            num_context_chunks=len(chunks),
            retrieval_scores=rag_resp.retrieval_scores,
            processing_time_ms=round(duration_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/eval/results", response_model=EvalResultsResponse)
async def get_eval_results():
    """Returns saved eval run reports from eval/reports/runs."""
    reports_dir = Path("eval/reports/runs")
    runs = []
    if reports_dir.exists():
        for report_file in sorted(reports_dir.glob("*.json"), reverse=True)[:10]:
            try:
                runs.append(json.loads(report_file.read_text(encoding="utf-8")))
            except Exception:
                pass

    latest_comp = None
    ablation_file = Path("eval/reports/ablation_report.json")
    if ablation_file.exists():
        try:
            latest_comp = json.loads(ablation_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return EvalResultsResponse(runs=runs, latest_comparison=latest_comp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
