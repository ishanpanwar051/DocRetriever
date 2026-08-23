"""
src/utils/memory.py — RAM Management for 8GB constraint

WHY this module exists:
  - llama3.2:3b      ~2.0 GB RAM
  - nomic-embed-text ~0.3 GB RAM
  - bge-reranker-base ~0.3 GB RAM
  Sequential model loading via keep_alive=0 + explicit unloads.
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — memory monitoring disabled.")


def get_ram_usage_gb() -> dict:
    if not PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}

    mem = psutil.virtual_memory()
    return {
        "total_gb":     round(mem.total / 1e9, 2),
        "used_gb":      round(mem.used / 1e9, 2),
        "available_gb": round(mem.available / 1e9, 2),
        "percent":      mem.percent,
    }


def log_memory(label: str = "") -> dict:
    stats = get_ram_usage_gb()
    if "error" not in stats:
        logger.info(
            f"[RAM] {label} | "
            f"Used: {stats['used_gb']}GB / {stats['total_gb']}GB "
            f"({stats['percent']}%) | "
            f"Available: {stats['available_gb']}GB"
        )
    return stats


class OllamaModelManager:
    """
    Manages Ollama model lifecycle for RAM-constrained environments.
    Supports both instance and classmethod usage.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def list_loaded(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.base_url}/api/ps", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list loaded Ollama models: {e}")
            return []

    def unload(self, model_name: str) -> bool:
        logger.info(f"[RAM] Unloading Ollama model: {model_name}")
        ram_before = log_memory(f"before unload {model_name}")

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            ram_after = log_memory(f"after unload {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload {model_name}: {e}")
            return False

    def ensure_only(self, needed_model: str) -> None:
        loaded = self.list_loaded()
        for model in loaded:
            if model != needed_model and needed_model not in model:
                self.unload(model)

        if loaded:
            time.sleep(1)

        ram = get_ram_usage_gb()
        if PSUTIL_AVAILABLE and "error" not in ram:
            if ram["available_gb"] < 3.0:
                logger.warning(
                    f"⚠️ Low RAM ({ram['available_gb']}GB free). "
                    f"Close other apps to prevent OOM."
                )

    @classmethod
    def ensure_model(cls, needed_model: str, base_url: Optional[str] = None) -> None:
        """Classmethod helper for quick one-line calls."""
        mgr = cls(base_url)
        mgr.ensure_only(needed_model)


@contextmanager
def managed_model(model_name: str, ollama_base_url: Optional[str] = None):
    manager = OllamaModelManager(ollama_base_url)
    manager.ensure_only(model_name)
    try:
        yield manager
    finally:
        manager.unload(model_name)
