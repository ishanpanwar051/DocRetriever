-- DocRetriever — PostgreSQL Initialization Script
-- Runs automatically on first container start
-- WHY separate init file: keeps schema version-controlled + reproducible

-- Step 1: Enable pgvector extension
-- WHY pgvector: native vector similarity search in Postgres = no extra infra (no Pinecone/Weaviate)
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Main chunks table
-- WHY this schema: covers all 4 retrieval strategies from one table
CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    source_file     TEXT NOT NULL,          -- e.g. "docs/en/docs/tutorial/first-steps.md"
    section_title   TEXT,                   -- heading under which chunk falls
    chunk_index     INTEGER NOT NULL,       -- position within source file (for debugging)
    content         TEXT NOT NULL,          -- raw text of the chunk
    token_count     INTEGER,                -- approximate token count (for monitoring)
    
    -- Strategy 1 & 2: vector similarity search
    embedding       vector(768),            -- nomic-embed-text output dim = 768
    
    -- Strategy 3: hybrid search (tsvector for BM25-like keyword search)
    content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    -- Metadata for filtering (Phase D+)
    chunk_strategy  TEXT DEFAULT 'simple',  -- which chunking strategy created this chunk
    metadata        JSONB DEFAULT '{}',     -- extensible metadata (page_url, section, etc.)
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Step 3: Indexes
-- WHY HNSW over IVFFlat: HNSW = better recall@k, no training step needed
-- ef_construction=128 = good quality/speed tradeoff for ~10k chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- WHY GIN for tsvector: GIN = inverted index, fast for full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
    ON document_chunks
    USING GIN (content_tsv);

-- Strategy filter index
CREATE INDEX IF NOT EXISTS idx_chunks_strategy
    ON document_chunks (chunk_strategy);

-- Source file index (for metadata filtering)
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON document_chunks (source_file);

-- Step 4: Experiment tracking table
-- WHY: reproducible results need experiment metadata stored alongside results
CREATE TABLE IF NOT EXISTS eval_runs (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT UNIQUE NOT NULL,   -- e.g. "simple_500t_50o_2024-01-15T10:30:00"
    strategy        TEXT NOT NULL,
    parameters      JSONB NOT NULL,         -- chunk_size, overlap, top_k, alpha, etc.
    metrics         JSONB,                  -- RAGAS + custom metrics results
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Verify setup
DO $$
BEGIN
    RAISE NOTICE 'DocRetriever DB initialized successfully';
    RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
    RAISE NOTICE 'Tables created: document_chunks, eval_runs';
END $$;
