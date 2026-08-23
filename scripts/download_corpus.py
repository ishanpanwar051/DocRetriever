"""
scripts/download_corpus.py — Download FastAPI docs (.md only) from GitHub

WHY GitHub API (not git clone)?
  - git clone downloads entire repo (~large). We only need .md files.
  - GitHub Trees API: list all files recursively, then download selectively.
  - No git required on the machine. Works with just Python + requests.

WHY FastAPI docs as corpus?
  - Open-source, legally free to use.
  - Well-structured: headings, code blocks, cross-references.
  - ~100+ .md files = sufficient for meaningful eval.
  - Familiar to AI/ML interviewers (they use FastAPI).

USAGE:
  python scripts/download_corpus.py
  python scripts/download_corpus.py --output corpus/fastapi_docs
  python scripts/download_corpus.py --branch master --max-files 50

Interview note: "I used the GitHub Trees API for selective download —
demonstrates understanding of API pagination and selective data fetching."
"""

import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GITHUB_API = "https://api.github.com"
REPO_OWNER = "tiangolo"
REPO_NAME = "fastapi"
BRANCH = "master"
DOCS_PREFIX = "docs/en/docs"           # only download from this subtree
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"
RATE_LIMIT_DELAY = 0.5                 # seconds between requests (GitHub rate limit: 60/hr unauthenticated)


def get_all_md_files(branch: str = BRANCH) -> list[dict]:
    """
    Uses GitHub Git Trees API to list all files recursively.
    Returns list of {path, size, download_url} for .md files under DOCS_PREFIX.
    
    WHY recursive=1: single API call instead of traversing directory by directory.
    """
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/{branch}?recursive=1"
    logger.info(f"Fetching file tree from: {url}")

    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 403:
            logger.error("GitHub rate limit hit. Wait 60 min or add GITHUB_TOKEN.")
            logger.error("Add to .env: GITHUB_TOKEN=ghp_xxxx (free, no scope needed)")
            raise
        raise

    data = resp.json()

    if data.get("truncated"):
        logger.warning("⚠️  GitHub tree response truncated — some files may be missing.")

    # Filter: only .md files under DOCS_PREFIX
    md_files = [
        item for item in data.get("tree", [])
        if (
            item["type"] == "blob"
            and item["path"].startswith(DOCS_PREFIX)
            and item["path"].endswith(".md")
        )
    ]

    logger.info(f"Found {len(md_files)} .md files under {DOCS_PREFIX}/")
    return md_files


def download_file(path: str, output_dir: Path, branch: str = BRANCH) -> bool:
    """
    Download a single file from GitHub raw content.
    Preserves directory structure relative to DOCS_PREFIX.
    """
    raw_url = f"{RAW_BASE}/{path}"

    # Relative path: strip DOCS_PREFIX so output is cleaner
    # e.g. "docs/en/docs/tutorial/first-steps.md" → "tutorial/first-steps.md"
    rel_path = path[len(DOCS_PREFIX):].lstrip("/")
    local_path = output_dir / rel_path

    # Skip if already downloaded (idempotent)
    if local_path.exists():
        return True

    # Create parent directories
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(raw_url, timeout=15)
        resp.raise_for_status()
        local_path.write_text(resp.text, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"Failed to download {path}: {e}")
        return False


def save_manifest(files: list[dict], output_dir: Path, stats: dict) -> None:
    """
    Save a manifest JSON with download metadata.
    WHY manifest: reproducibility — know exactly what corpus version was used.
    """
    manifest = {
        "downloaded_at": datetime.now().isoformat(),
        "source": f"github.com/{REPO_OWNER}/{REPO_NAME}",
        "branch": BRANCH,
        "docs_prefix": DOCS_PREFIX,
        "total_files": stats["total"],
        "downloaded": stats["success"],
        "failed": stats["failed"],
        "files": [
            {
                "github_path": f["path"],
                "local_path": f["path"][len(DOCS_PREFIX):].lstrip("/"),
                "size_bytes": f.get("size", 0),
            }
            for f in files
        ],
    }
    manifest_path = output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest saved: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FastAPI docs .md files from GitHub"
    )
    parser.add_argument(
        "--output", default="corpus/fastapi_docs",
        help="Output directory (default: corpus/fastapi_docs)"
    )
    parser.add_argument(
        "--branch", default=BRANCH,
        help=f"GitHub branch (default: {BRANCH})"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of files (for testing)"
    )
    parser.add_argument(
        "--delay", type=float, default=RATE_LIMIT_DELAY,
        help=f"Delay between requests in seconds (default: {RATE_LIMIT_DELAY})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DocRetriever — FastAPI Corpus Downloader")
    print(f"  Output: {output_dir.absolute()}")
    print(f"  Source: github.com/{REPO_OWNER}/{REPO_NAME} [{args.branch}]")
    print(f"{'='*60}\n")

    # Step 1: Get file list
    logger.info("Step 1: Fetching file list from GitHub API...")
    md_files = get_all_md_files(branch=args.branch)

    if args.max_files:
        md_files = md_files[:args.max_files]
        logger.info(f"Limited to {args.max_files} files (--max-files flag)")

    # Step 2: Download each file
    logger.info(f"Step 2: Downloading {len(md_files)} .md files...")
    stats = {"total": len(md_files), "success": 0, "failed": 0, "skipped": 0}

    with tqdm(md_files, desc="Downloading", unit="file") as pbar:
        for file_info in pbar:
            path = file_info["path"]
            rel = path[len(DOCS_PREFIX):].lstrip("/")
            pbar.set_postfix({"file": rel[:40]})

            local = output_dir / rel
            if local.exists():
                stats["skipped"] += 1
                continue

            success = download_file(path, output_dir, branch=args.branch)
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1

            time.sleep(args.delay)

    # Step 3: Save manifest
    save_manifest(md_files, output_dir, stats)

    # Step 4: Report
    print(f"\n{'='*60}")
    print(f"  ✅ Download Complete!")
    print(f"     Total files:   {stats['total']}")
    print(f"     Downloaded:    {stats['success']}")
    print(f"     Skipped:       {stats['skipped']} (already existed)")
    print(f"     Failed:        {stats['failed']}")
    print(f"     Output dir:    {output_dir.absolute()}")

    # Show directory tree summary
    all_md = list(output_dir.rglob("*.md"))
    total_size_kb = sum(f.stat().st_size for f in all_md) / 1024
    print(f"     Total .md files on disk: {len(all_md)}")
    print(f"     Total corpus size: {total_size_kb:.1f} KB")
    print(f"{'='*60}\n")

    if stats["failed"] > 0:
        logger.warning(f"{stats['failed']} files failed to download. "
                       f"Re-run the script — it's idempotent (skips existing files).")


if __name__ == "__main__":
    main()
