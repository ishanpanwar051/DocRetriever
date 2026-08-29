"""
config/settings.py — Centralized configuration using Pydantic Settings v2

WHY Pydantic Settings (not raw os.environ)?
- Type validation at startup (catches missing vars immediately, not at runtime)
- Auto-loads from .env file
- Provides IDE autocompletion for all config vars
- Interview answer: "fail fast on config errors, not silently at 3am"
"""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_user: str = Field(default="docretriever")
    postgres_password: str = Field(default="docretriever_pass")
    postgres_db: str = Field(default="docretriever_db")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    @computed_field
    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible PostgreSQL connection string."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Groq Cloud API (LLM Generation) ───────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_llm_model: str = Field(default="llama-3.1-8b-instant")

    # ── Embeddings (sentence-transformers, runs locally in CPU) ────────────────
    embed_model: str = Field(default="all-MiniLM-L6-v2")   # 384-dim, fast on CPU
    embedding_dim: int = Field(default=384)  # all-MiniLM-L6-v2 output dim

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    reranker_device: str = Field(default="cpu")

    # ── Retrieval Defaults ────────────────────────────────────────────────────
    default_top_k: int = Field(default=5)
    default_strategy: str = Field(default="simple")

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_judge_model: str = Field(default="llama-3.1-8b-instant")
    eval_dataset_path: str = Field(default="eval/data/qa_pairs.jsonl")
    eval_reports_dir: str = Field(default="eval/reports")

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_dir: str = Field(default="corpus/fastapi_docs")


# Singleton — import this everywhere
settings = Settings()


if __name__ == "__main__":
    # Quick test: python -m config.settings
    print("✅ Settings loaded:")
    print(f"  DB URL:      {settings.database_url}")
    print(f"  Embed model: {settings.embed_model}")
    print(f"  LLM model:   {settings.groq_llm_model}")
    print(f"  Reranker:    {settings.reranker_model} on {settings.reranker_device}")
