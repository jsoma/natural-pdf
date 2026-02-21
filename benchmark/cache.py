"""
Disk-based caching for LLM API responses.

Cache structure: llm-cache/{pdf_name}/{model}.json

Validates cache entries by checking:
- PDF content hash (ensures same file)
- Prompt hash (ensures same extraction prompt)
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hash_file(path: str | Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_string(text: str) -> str:
    """Compute SHA256 hash of string."""
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class CachedResponse:
    """A cached LLM response."""

    model: str
    provider: str
    pdf_path: str
    pdf_hash: str  # SHA256 of PDF content
    prompt_hash: str  # SHA256 of prompt
    raw_response: str
    parsed_data: list[dict[str, Any]]
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cached_at: str


class ResponseCache:
    """
    Disk-based cache for LLM responses.

    Structure: {cache_dir}/{pdf_name}/{model}.json
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0}

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for use as filename."""
        return model.replace("/", "_").replace(":", "_").replace("@", "_")

    def _cache_path(self, pdf_name: str, model: str) -> Path:
        """Get path for a cache entry: {cache_dir}/{pdf_name}/{model}.json"""
        pdf_dir = self.cache_dir / pdf_name
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return pdf_dir / f"{self._normalize_model_name(model)}.json"

    def get(
        self,
        pdf_name: str,
        pdf_path: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> Optional[CachedResponse]:
        """
        Get cached response if available and valid.

        Validates that PDF content and prompt haven't changed.
        Returns None if not cached or validation fails.
        """
        cache_path = self._cache_path(pdf_name, model)

        if not cache_path.exists():
            self._stats["misses"] += 1
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Validate PDF hash
            current_pdf_hash = _hash_file(pdf_path)
            if data.get("pdf_hash") != current_pdf_hash:
                logger.info(f"Cache miss for {pdf_name}/{model}: PDF content changed")
                self._stats["misses"] += 1
                return None

            # Validate prompt hash
            current_prompt_hash = _hash_string(prompt)
            if data.get("prompt_hash") != current_prompt_hash:
                logger.info(f"Cache miss for {pdf_name}/{model}: prompt changed")
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return CachedResponse(**data)

        except (json.JSONDecodeError, TypeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Cache read error for {pdf_name}/{model}: {e}")
            cache_path.unlink(missing_ok=True)
            self._stats["misses"] += 1
            return None

    def set(
        self,
        pdf_name: str,
        pdf_path: str,
        prompt: str,
        model: str,
        provider: str,
        raw_response: str,
        parsed_data: list[dict[str, Any]],
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: int = 0,
    ) -> CachedResponse:
        """Cache a response with validation hashes."""
        cached = CachedResponse(
            model=model,
            provider=provider,
            pdf_path=pdf_path,
            pdf_hash=_hash_file(pdf_path),
            prompt_hash=_hash_string(prompt),
            raw_response=raw_response,
            parsed_data=parsed_data,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cached_at=datetime.now().isoformat(),
        )

        cache_path = self._cache_path(pdf_name, model)

        # Atomic write: write to temp file, then rename
        tmp_path = cache_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(asdict(cached), f, indent=2)
            tmp_path.replace(cache_path)
        except Exception as e:
            logger.error(f"Failed to write cache {pdf_name}/{model}: {e}")
            tmp_path.unlink(missing_ok=True)
            raise

        return cached

    def invalidate(self, pdf_name: str, model: str) -> bool:
        """Remove a cache entry. Returns True if entry existed."""
        cache_path = self._cache_path(pdf_name, model)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns count of entries removed."""
        count = 0
        for path in self.cache_dir.rglob("*.json"):
            path.unlink()
            count += 1
        return count

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total_entries = sum(1 for _ in self.cache_dir.rglob("*.json"))
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total_entries": total_entries,
            "hit_rate": (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            ),
        }

    def size_bytes(self) -> int:
        """Get total cache size in bytes."""
        return sum(p.stat().st_size for p in self.cache_dir.rglob("*.json"))

    def list_entries(self, model: Optional[str] = None) -> list[CachedResponse]:
        """List all cached entries, optionally filtered by model."""
        entries = []
        for path in self.cache_dir.rglob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                entry = CachedResponse(**data)
                if model is None or entry.model == model:
                    entries.append(entry)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return entries
