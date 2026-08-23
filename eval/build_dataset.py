"""
eval/build_dataset.py — Semi-automated QA dataset builder

WHY semi-automated (not fully manual):
- Manual: slow, inconsistent, hard to scale
- Fully automated (LLM-only): hallucinated questions, no quality control
- Semi-auto: script extracts candidates from docs structure → human reviews/selects
  Best of both: speed + quality
  Interview: 'I generated 100+ candidates, manually curated 40, ensuring diversity'
"""

import json
import argparse
import re
from pathlib import Path
from config.settings import settings


def extract_candidates_from_docs(corpus_dir: str = settings.corpus_dir) -> list[dict]:
    """Extract question candidates from FastAPI docs headings and code blocks."""
    candidates = []
    base_path = Path(corpus_dir)
    
    if not base_path.exists():
        print(f"⚠️ Corpus directory '{corpus_dir}' not found. Run 'python scripts/download_corpus.py' first.")
        return candidates
        
    for md_file in sorted(base_path.rglob("*.md")):
        rel_path = str(md_file.relative_to(base_path)).replace("\\", "/")
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
            
        # 1. Parse H2/H3 headings
        headings = re.findall(r'^(#{2,3})\s+(.+)$', content, re.MULTILINE)
        for _, title in headings:
            clean_title = title.strip()
            if len(clean_title) > 3 and not clean_title.startswith("!"):
                question_draft = f"How does {clean_title} work in FastAPI?"
                candidates.append({
                    "question_draft": question_draft,
                    "source_file": rel_path,
                    "section": clean_title,
                    "type": "extracted_heading",
                })
            
        # 2. Find code blocks usage
        code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
        for block in code_blocks:
            if '@app.' in block and 'def ' in block:
                # Extract route handler function name if possible
                fn_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', block)
                fn_name = fn_match.group(1) if fn_match else "endpoint"
                candidates.append({
                    "question_draft": f"How do you implement the `{fn_name}` pattern in FastAPI?",
                    "source_file": rel_path,
                    "section": f"Code Example ({fn_name})",
                    "type": "extracted_code",
                })
                
    return candidates


def save_candidates(candidates: list[dict], output: str = "eval/data/candidates.jsonl"):
    """Save candidate QA drafts for manual curation."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, cand in enumerate(candidates, 1):
            cand["id"] = idx
            f.write(json.dumps(cand) + "\n")
    print(f"✓ Saved {len(candidates)} candidate questions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract QA candidates from corpus")
    parser.add_argument("--corpus-dir", default=settings.corpus_dir, help="Corpus root directory")
    parser.add_argument("--output", default="eval/data/candidates.jsonl", help="Output candidates JSONL path")
    args = parser.parse_args()

    print(f"\n📂 Extracting QA candidates from: {args.corpus_dir}")
    candidates = extract_candidates_from_docs(args.corpus_dir)
    print(f"Extracted {len(candidates)} candidate questions.")
    if candidates:
        save_candidates(candidates, output=args.output)
        print("💡 Review eval/data/candidates.jsonl to curate final evaluation QA pairs.\n")
