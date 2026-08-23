"""
src/ingestion/ingest.py — Ingestion Pipeline for DocRetriever

WHY this pipeline design:
- MarkdownParser extracts semantic heading boundaries (# H1, ## H2)
- SimpleChunker (500 tokens, 50 overlap) or SemanticChunker (cosine drop)
- OllamaEmbedder embeds in batches of 32 to prevent 8GB RAM saturation
- Bulk writes to PostgreSQL pgvector table document_chunks
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
from sqlalchemy import delete

from config.settings import settings
from src.db.connection import get_db
from src.db.models import DocumentChunk
from src.ingestion.markdown_parser import MarkdownParser
from src.ingestion.chunker import SimpleChunker, SemanticChunker
from src.ingestion.embedder import OllamaEmbedder
from src.utils.memory import OllamaModelManager


def ingest_corpus(
    corpus_dir: str = settings.corpus_dir,
    strategy: str = "simple",
    chunk_size: int = 500,
    overlap: int = 50,
    threshold: float = 0.3,
    batch_size: int = 32,
    clear_existing: bool = False,
) -> tuple[int, int]:
    """
    Main ingestion pipeline: parse -> chunk -> embed -> store.
    Returns (files_processed, chunks_created).
    """
    # 1. RAM check: ensure only embedder is loaded
    try:
        mgr = OllamaModelManager(settings.ollama_base_url)
        mgr.ensure_only(settings.ollama_embed_model)
    except Exception as e:
        print(f"Warning: Could not manage Ollama memory (is Ollama running?): {e}")

    # 2. Load all .md files
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"Corpus directory not found: {corpus_path}. Running download...")
        import subprocess
        subprocess.run(["python", "scripts/download_corpus.py"], check=False)

    md_files = list(corpus_path.rglob("*.md"))
    if not md_files:
        raise ValueError(f"No .md files found in {corpus_path}. Run scripts/download_corpus.py first!")

    print(f"\n📂 Found {len(md_files)} Markdown files in {corpus_path}.")

    # 3. Parse each with MarkdownParser
    parser = MarkdownParser()
    all_sections = []
    for md_file in tqdm(md_files, desc="Parsing Markdown"):
        sections = parser.parse_file(md_file)
        for sec in sections:
            sec["source_file"] = str(md_file.relative_to(corpus_path)).replace("\\", "/")
        all_sections.extend(sections)

    print(f"📄 Parsed {len(all_sections)} semantic sections.")

    # 4. Chunk each section
    embedder = OllamaEmbedder(model=settings.ollama_embed_model, batch_size=batch_size)

    if strategy == "semantic":
        chunker = SemanticChunker(embed_fn=embedder.embed_texts, threshold=threshold)
    else:
        chunker = SimpleChunker(chunk_size=chunk_size, overlap=overlap)

    all_chunks_data = []
    for section in tqdm(all_sections, desc=f"Chunking ({strategy})"):
        if strategy == "semantic":
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section["content"]) if s.strip()]
            chunks = chunker.chunk(sentences) if sentences else [section["content"]]
        else:
            chunks = chunker.chunk(section["content"])

        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                all_chunks_data.append({
                    "source_file": section["source_file"],
                    "section_title": section.get("section_title"),
                    "chunk_index": i,
                    "content": chunk_text.strip(),
                    "chunk_strategy": strategy,
                })

    print(f"🧩 Created {len(all_chunks_data)} chunks.")

    # 5. Embed all chunks (batched)
    chunk_texts = [c["content"] for c in all_chunks_data]
    print("🧠 Generating embeddings with nomic-embed-text (batch_size=32)...")
    embeddings = embedder.embed_texts(chunk_texts)

    for chunk_data, emb in zip(all_chunks_data, embeddings):
        chunk_data["embedding"] = emb
        chunk_data["token_count"] = len(chunk_data["content"].split())

    # 6. Store in PostgreSQL
    with get_db() as db:
        if clear_existing:
            db.execute(delete(DocumentChunk).where(DocumentChunk.chunk_strategy == strategy))
            db.commit()
            print(f"🗑️ Cleared existing chunks for strategy '{strategy}'.")

        db_chunks = [DocumentChunk(**data) for data in all_chunks_data]
        db.add_all(db_chunks)
        db.commit()

    print(f"\n✅ Ingestion complete! Stored {len(all_chunks_data)} chunks in PostgreSQL pgvector.")
    return len(md_files), len(all_chunks_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Markdown corpus into pgvector")
    parser.add_argument("--corpus-dir", default=settings.corpus_dir, help="Corpus root directory")
    parser.add_argument("--strategy", choices=["simple", "semantic"], default="simple")
    parser.add_argument("--chunk-size", type=int, default=500, help="Tokens per chunk (for simple)")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap (for simple)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Cosine distance drop threshold (for semantic)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clear", action="store_true", help="Clear existing strategy chunks before ingest")
    args = parser.parse_args()

    ingest_corpus(
        corpus_dir=args.corpus_dir,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        threshold=args.threshold,
        batch_size=args.batch_size,
        clear_existing=args.clear,
    )
