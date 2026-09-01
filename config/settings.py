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

    # ── Multi-Provider LLM Configuration ───────────────────────────────────────
    default_llm_provider: str = Field(default="groq")  # 'groq', 'openai', 'anthropic', 'ollama'
    
    # Groq Cloud API
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_llm_model: str = Field(default="llama-3.1-8b-instant")

    # OpenAI API
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_llm_model: str = Field(default="gpt-4o-mini")

    # Anthropic API
    anthropic_api_key: str = Field(default="")
    anthropic_llm_model: str = Field(default="claude-3-haiku-20240307")

    # Local Ollama (Offline / Zero-Cost)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="llama3.2:3b")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    ollama_keep_alive: int = Field(default=0)

    # ── Embeddings (sentence-transformers, runs locally on CPU) ───────────────
    embed_model: str = Field(default="all-MiniLM-L6-v2")   # 384-dim, fast on CPU
    embedding_dim: int = Field(default=384)  # all-MiniLM-L6-v2 output dim

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    reranker_device: str = Field(default="cpu")

    # ── Retrieval Defaults & Parameters ───────────────────────────────────────
    default_top_k: int = Field(default=5)
    candidate_top_k: int = Field(default=20)  # Candidates for MMR / Reranking
    default_strategy: str = Field(default="rerank")
    hybrid_alpha: float = Field(default=0.5)  # 0.0 (Pure Sparse) -> 1.0 (Pure Dense)
    mmr_lambda: float = Field(default=0.7)    # 0.0 (Max Diversity) -> 1.0 (Max Relevance)
    similarity_threshold: float = Field(default=0.0) # Optional confidence threshold

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_judge_model: str = Field(default="llama-3.1-8b-instant")
    eval_dataset_path: str = Field(default="eval/data/qa_pairs.jsonl")
    eval_reports_dir: str = Field(default="eval/reports")

    # ── API & Security ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="*")
    log_level: str = Field(default="INFO")
    enable_telemetry: bool = Field(default=True)

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_dir: str = Field(default="corpus/fastapi_docs")


# Singleton — import this everywhere
settings = Settings()


if __name__ == "__main__":
    # Quick test: python -m config.settings
    print("✅ Settings loaded:")
    print(f"  DB URL:           {settings.database_url}")
    print(f"  Embed model:      {settings.embed_model} (dim={settings.embedding_dim})")
    print(f"  Default Provider: {settings.default_llm_provider}")
    print(f"  Reranker:         {settings.reranker_model} on {settings.reranker_device}")
