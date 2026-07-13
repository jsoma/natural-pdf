"""OCR result caching — avoids redundant engine calls on the same page.

Results are cached in the user's platform cache directory
(``~/.cache/natural-pdf/ocr/`` on Linux/macOS, AppData on Windows).
Cache entries are keyed on ``(pdf_path, file_mtime, file_size,
page_index, rendered crop, engine, languages, resolution, ...)``, so they
auto-invalidate when the PDF changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_UNSTABLE = object()
_DUPLICATED_RENDER_KEYS = frozenset({"_ocr_exclusion_bboxes"})


def _stable_json_value(value: Any) -> Any:
    """Return a canonical JSON-like value, or ``_UNSTABLE`` if unsafe."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSTABLE
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            normalized = _stable_json_value(item)
            if normalized is _UNSTABLE:
                return _UNSTABLE
            items.append(normalized)
        return items
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return _UNSTABLE
        normalized_dict: Dict[str, Any] = {}
        for key in sorted(value):
            normalized = _stable_json_value(value[key])
            if normalized is _UNSTABLE:
                return _UNSTABLE
            normalized_dict[key] = normalized
        return normalized_dict
    return _UNSTABLE


def compute_render_kwargs_cache_key(render_kwargs: Dict[str, Any]) -> Optional[str]:
    """Hash all stably representable render inputs used for OCR rasterization.

    Internal exclusion boxes are keyed separately by their normalized geometry.
    Any other non-JSON-like custom value disables persistent result caching so
    cache correctness never depends on an unstable object representation.
    """
    filtered = {
        key: value for key, value in render_kwargs.items() if key not in _DUPLICATED_RENDER_KEYS
    }
    normalized = _stable_json_value(filtered)
    if normalized is _UNSTABLE:
        return None
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _client_cache_namespace(client: Any) -> Optional[str]:
    """Return an explicitly supplied, non-secret identity for a VLM client.

    Remote clients can target different deployments even when their Python type
    and requested model name match.  Never derive an identity from ``repr()``,
    URLs, or client attributes: those are often unstable and can contain API
    credentials.  A client that wants persistent OCR-result caching must opt in
    by exposing a non-empty ``natural_pdf_cache_namespace`` string.
    """
    namespace = getattr(client, "natural_pdf_cache_namespace", None)
    return namespace.strip() if isinstance(namespace, str) and namespace.strip() else None


def resolve_ocr_cache_identity(
    *,
    engine_name: str,
    device: Optional[str],
    model: Optional[str],
    client: Any,
) -> Optional[Dict[str, str]]:
    """Resolve the result-affecting backend identity for persistent OCR cache.

    ``None`` means that a safe persistent result key cannot be constructed. In
    particular, remote/custom clients need to opt in with
    ``natural_pdf_cache_namespace``. This deliberately favors rerunning OCR
    over returning a result produced by another deployment or callable.
    """
    from natural_pdf.core.vlm_client import get_default_client
    from natural_pdf.ocr.unified_dispatch import _is_apple_silicon, get_registry
    from natural_pdf.utils.option_validation import resolve_auto_device

    engine_key = engine_name.strip().lower()
    entry = get_registry().get(engine_key)
    if entry is None:
        return None

    requested_device = device or "auto"
    effective_device = (
        resolve_auto_device() if requested_device == "auto" else str(requested_device)
    )
    engine_type = entry.engine_type
    effective_model = model
    uses_vlm = engine_type in {"vlm_generic", "vlm_shorthand"}

    if engine_type == "auto_platform":
        if _is_apple_silicon():
            uses_vlm = True
            effective_model = model or (
                entry.model_resolver() if entry.model_resolver is not None else None
            )
        else:
            engine_type = "classic"
    elif engine_type == "vlm_shorthand" and effective_model is None:
        effective_model = entry.model_resolver() if entry.model_resolver is not None else None

    # Every engine, including built-ins, carries an explicit namespace on its
    # registration. This prevents a custom re-registration under a built-in
    # name from inheriting the built-in's persistent result cache.
    namespace = getattr(entry, "cache_namespace", None)
    if not isinstance(namespace, str) or not namespace.strip():
        return None
    engine_identity = namespace.strip()

    identity = {
        "engine": engine_identity,
        "engine_type": engine_type,
        "device": effective_device,
        "model": effective_model or "",
    }

    if not uses_vlm:
        return identity

    default_client, default_model = get_default_client()
    effective_client = client if client is not None else default_client
    if effective_model is None:
        effective_model = default_model
        identity["model"] = effective_model or ""

    if effective_client is None:
        identity["client"] = "local"
        return identity

    client_namespace = _client_cache_namespace(effective_client)
    if client_namespace is None:
        return None
    identity["client"] = client_namespace
    return identity


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
    options_init_key: str = "",
    apply_exclusions: bool = True,
    min_confidence: Optional[float] = None,
    options_cache_key: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    layout: Optional[bool | str] = None,
    preserve_markup: bool = False,
    crop_bbox: Optional[Tuple[float, float, float, float]] = None,
    exclusion_geometry_key: str = "",
    render_kwargs_cache_key: str = "",
    execution_identity: Optional[Dict[str, str]] = None,
) -> str:
    """Return a SHA-256 hex digest for the given OCR parameters."""
    crop_key = ""
    if crop_bbox is not None:
        crop_key = ",".join(f"{float(coord):.6f}" for coord in crop_bbox)
    options_key = options_cache_key if options_cache_key is not None else options_init_key

    raw = json.dumps(
        {
            "pdf_path": pdf_path,
            "file_mtime_ns": file_mtime_ns,
            "file_size": file_size,
            "page_index": page_index,
            "crop": crop_key,
            "engine": engine_name,
            "languages": list(languages),
            "resolution": resolution,
            "detect_only": detect_only,
            "device": device,
            "min_confidence": min_confidence,
            "options": options_key,
            "apply_exclusions": apply_exclusions,
            "model": model,
            "prompt": prompt,
            "instructions": instructions,
            "max_new_tokens": max_new_tokens,
            "layout": layout,
            "preserve_markup": preserve_markup,
            "exclusion_geometry": exclusion_geometry_key,
            "render_kwargs": render_kwargs_cache_key,
            "execution_identity": execution_identity or {},
        },
        sort_keys=True,
        separators=(",", ":"),
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
            # Migrate entries written by older versions before reading them.
            # Cache data is disposable, so fail closed if its permissions
            # cannot be tightened.
            self._ensure_private_directory(self._dir)
            self._ensure_private_directory(path.parent)
            path.chmod(0o600)
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

    def delete(self, cache_key: str) -> bool:
        """Remove one cached entry. Returns ``True`` if an entry was removed."""
        path = self._key_path(cache_key)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            logger.debug("OCR cache delete failed for %s", cache_key, exc_info=True)
            return False

    def put(
        self,
        cache_key: str,
        result: "Any",
        engine_name: str,
        page_index: int,
    ) -> None:
        """Write an ``OCRRunResult`` to the cache."""
        path = self._key_path(cache_key)
        tmp = path.with_suffix(".tmp")
        try:
            self._ensure_private_directory(self._dir)
            self._ensure_private_directory(path.parent)
            data = {
                "version": _CACHE_VERSION,
                "created_at": time.time(),
                "engine_name": engine_name,
                "page_index": page_index,
                "image_size": list(result.image_size),
                "engine_type": getattr(result, "engine_type", "classic"),
                "results": _serializable_results(result.results),
            }
            self._write_private_json(tmp, data)
            tmp.replace(path)  # atomic on same filesystem
            path.chmod(0o600)
        except Exception:
            logger.debug("OCR cache write failed for %s", cache_key, exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

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

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)

    @staticmethod
    def _write_private_json(path: Path, data: Dict[str, Any]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with handle:
            json.dump(data, handle, separators=(",", ":"))

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
