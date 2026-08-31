import pytest
from src.ingestion.chunker import SimpleChunker, SemanticChunker, cosine_sim
from src.ingestion.markdown_parser import MarkdownParser
from pathlib import Path
import tempfile


def test_simple_chunker_small_text():
    chunker = SimpleChunker(chunk_size=100, overlap=20)
    text = "FastAPI is a modern, fast web framework for building APIs with Python."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "FastAPI" in chunks[0]


def test_simple_chunker_overlap():
    chunker = SimpleChunker(chunk_size=20, overlap=5)
    text = (
        "FastAPI is a modern, fast web framework for building APIs with Python 3.8+ "
        "based on standard Python type hints. The key features are fast to code, "
        "fewer bugs, intuitive, easy, and robust."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert len(chunks[0]) > 0
    assert len(chunks[1]) > 0


def test_cosine_sim_identical():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert abs(cosine_sim(v1, v2) - 1.0) < 1e-5


def test_cosine_sim_orthogonal():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert abs(cosine_sim(v1, v2) - 0.0) < 1e-5


def test_semantic_chunker_splits_on_low_similarity():
    def mock_embed(sentences):
        embeddings = []
        for i, s in enumerate(sentences):
            if i < 2:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return embeddings

    chunker = SemanticChunker(embed_fn=mock_embed, threshold=0.5, max_chunk_tokens=100)
    sentences = ["Sentence 1.", "Sentence 2.", "Completely different topic 3.", "Topic 4."]
    chunks = chunker.chunk(sentences)
    assert len(chunks) == 2
    assert "Sentence 1. Sentence 2." in chunks[0]
    assert "Completely different topic 3. Topic 4." in chunks[1]


def test_markdown_parser_code_block_protection():
    md_content = """# Main Header
This is intro text.

```python
# This is a comment inside code, not a header
def foo():
    pass
```

## Second Header
This is second section.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(md_content)
        temp_path = Path(f.name)

    parser = MarkdownParser()
    sections = parser.parse_file(temp_path)
    temp_path.unlink()

    assert len(sections) == 2
    assert sections[0]["section_title"] == "Main Header"
    assert "def foo():" in sections[0]["content"]
    assert sections[1]["section_title"] == "Second Header"
