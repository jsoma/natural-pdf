"""OCR result caching — avoids redundant engine calls on the same page.

Results are cached in the user's platform cache directory
(``~/.cache/natural-pdf/ocr/`` on Linux/macOS, AppData on Windows).
Cache entries are keyed on ``(pdf_path, file_mtime, file_size,
page_index, engine, languages, resolution, ...)``, so they
auto-invalidate when the PDF changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


def _default_cache_dir() -> Path:
    """Return the platform-appropriate cache directory."""
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("natural_pdf")) / "ocr"
    except ImportError:
        return Path.home() / ".cache" / "natural-pdf" / "ocr"


def compute_cache_key(
    pdf_path: str,
    file_mtime_ns: int,
    file_size: int,
    page_index: int,
    engine_name: str,
    languages: Tuple[str, ...],
    resolution: int,
    detect_only: bool,
    device: str,
    options_init_key: str,
    apply_exclusions: bool = True,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Return a SHA-256 hex digest for the given OCR parameters."""
    raw = "|".join(
        str(v)
        for v in (
            pdf_path,
            file_mtime_ns,
            file_size,
            page_index,
            engine_name,
            ",".join(languages),
            resolution,
            detect_only,
            device,
            options_init_key,
            apply_exclusions,
            model or "",
            prompt or "",
            instructions or "",
            max_new_tokens or "",
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class OCRCache:
    """Disk-backed cache for OCR results.

    Parameters
    ----------
    cache_dir : Path, optional
        Root directory for cache files.  Defaults to the platform cache dir.
    ttl_days : int
        Entries older than this are eligible for cleanup.
    max_size_mb : int
        Target maximum total size of cached files.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_days: int = 30,
        max_size_mb: int = 500,
    ):
        self._dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._ttl_seconds = ttl_days * 86400
        self._max_bytes = max_size_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, cache_key: str) -> "Optional[Any]":
        """Return a cached ``OCRRunResult`` or ``None`` on miss."""
        path = self._key_path(cache_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != _CACHE_VERSION:
                path.unlink(missing_ok=True)
                return None
            # Reconstruct OCRRunResult
            from natural_pdf.ocr.unified_dispatch import OCRRunResult

            return OCRRunResult(
                results=data["results"],
                image_size=tuple(data["image_size"]),
                engine_type=data.get("engine_type", "classic"),
            )
        except Exception:
            logger.debug("OCR cache read failed for %s", cache_key, exc_info=True)
            return None

    def put(
        self,
        cache_key: str,
        result: "Any",
        engine_name: str,
        page_index: int,
    ) -> None:
        """Write an ``OCRRunResult`` to the cache."""
        path = self._key_path(cache_key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": _CACHE_VERSION,
                "created_at": time.time(),
                "engine_name": engine_name,
                "page_index": page_index,
                "image_size": list(result.image_size),
                "engine_type": getattr(result, "engine_type", "classic"),
                "results": _serializable_results(result.results),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)  # atomic on same filesystem
        except Exception:
            logger.debug("OCR cache write failed for %s", cache_key, exc_info=True)

        # Lazy cleanup — once per process
        self._maybe_cleanup()

    def clear(self) -> int:
        """Remove **all** cached entries.  Returns number of files deleted."""
        count = 0
        if not self._dir.exists():
            return count
        for f in self._dir.rglob("*.json"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        # Remove empty subdirectories
        for d in sorted(self._dir.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass
        return count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key_path(self, cache_key: str) -> Path:
        return self._dir / cache_key[:2] / f"{cache_key}.json"

    _cleaned_dirs: set = set()

    def _maybe_cleanup(self) -> None:
        if self._dir in OCRCache._cleaned_dirs:
            return
        OCRCache._cleaned_dirs.add(self._dir)
        try:
            self._cleanup()
        except Exception:
            logger.debug("OCR cache cleanup failed", exc_info=True)

    def _cleanup(self) -> None:
        if not self._dir.exists():
            return
        now = time.time()
        entries: List[Tuple[Path, float, int]] = []
        for f in self._dir.rglob("*.json"):
            try:
                st = f.stat()
                entries.append((f, st.st_mtime, st.st_size))
            except OSError:
                continue

        # Phase 1: delete entries older than TTL
        remaining = []
        for fpath, mtime, size in entries:
            if now - mtime > self._ttl_seconds:
                try:
                    fpath.unlink()
                except OSError:
                    pass
            else:
                remaining.append((fpath, mtime, size))

        # Phase 2: if still over budget, delete oldest first
        total = sum(s for _, _, s in remaining)
        if total > self._max_bytes:
            remaining.sort(key=lambda x: x[1])  # oldest first
            for fpath, _mtime, size in remaining:
                if total <= self._max_bytes:
                    break
                try:
                    fpath.unlink()
                    total -= size
                except OSError:
                    pass


def _serializable_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure result dicts are JSON-serializable (convert tuples to lists, etc.)."""
    out = []
    for r in results:
        entry: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, tuple):
                entry[k] = list(v)
            else:
                entry[k] = v
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_cache: Optional[OCRCache] = None


def get_default_cache() -> OCRCache:
    """Return (and lazily create) the module-level OCR cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = OCRCache()
    return _default_cache


def set_default_cache(cache: Optional[OCRCache]) -> Optional[OCRCache]:
    """Replace the module-level OCR cache singleton.

    Pass an :class:`OCRCache` instance (e.g. one backed by a temp dir)
    to redirect all caching, or ``None`` to reset to the default.
    Returns the *previous* cache so callers can restore it.
    """
    global _default_cache
    previous = _default_cache
    _default_cache = cache
    return previous
