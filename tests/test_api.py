import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app
from src.retrieval.base import Chunk
from src.generation.generator import RAGResponse, SourceCitation

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "postgres" in data
    assert "groq_api" in data
    assert "corpus_files" in data


def test_eval_results_endpoint():
    response = client.get("/eval/results")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data


@patch("api.main.get_retriever")
@patch("api.main.RAGGenerator")
def test_ask_endpoint_mocked(mock_gen_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id=1, content="FastAPI test passage", source_file="tutorial/first-steps.md", section_title="First Steps", chunk_index=0, score=0.92)
    ]
    mock_get_retriever.return_value = mock_retriever

    mock_generator = MagicMock()
    mock_generator.generate.return_value = RAGResponse(
        answer="FastAPI is an async framework.",
        sources=[SourceCitation(source_file="tutorial/first-steps.md", section_title="First Steps")],
        strategy="simple",
        num_context_chunks=1,
        retrieval_scores=[0.92],
    )
    mock_gen_cls.return_value = mock_generator

    response = client.post("/ask", json={"question": "What is FastAPI?", "strategy": "simple", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI is an async framework."
    assert data["strategy"] == "simple"
    assert data["num_context_chunks"] == 1
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_file"] == "tutorial/first-steps.md"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "feedback" in data


def test_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_docs" in data


@patch("api.main.get_retriever")
@patch("api.main.RAGGenerator")
def test_ask_stream_endpoint(mock_gen_cls, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id=1, content="FastAPI test passage", source_file="tutorial/first-steps.md", section_title="First Steps", chunk_index=0, score=0.92)
    ]
    mock_get_retriever.return_value = mock_retriever

    mock_generator = MagicMock()
    def mock_stream_gen(*args, **kwargs):
        yield {"type": "metadata", "strategy": "simple", "sources": [], "num_chunks": 1}
        yield {"type": "token", "content": "FastAPI "}
        yield {"type": "token", "content": "is great."}
        yield {"type": "done", "full_answer": "FastAPI is great."}
    
    mock_generator.generate_stream = mock_stream_gen
    mock_gen_cls.return_value = mock_generator

    response = client.post("/ask/stream", json={"question": "What is FastAPI?", "strategy": "simple", "top_k": 5})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data: " in content
    assert "FastAPI" in content

