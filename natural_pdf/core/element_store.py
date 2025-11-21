from __future__ import annotations

import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set

Callback = Callable[[str], None]

logger = logging.getLogger(__name__)


class ElementStore:
    """Thread-safe container for page elements with invalidation callbacks."""

    def __init__(self) -> None:
        self._data: Dict[str, List[Any]] = {}
        self._lock = threading.RLock()
        self._version = 0
        self._loaded = False
        self._callbacks: DefaultDict[str, List[Callback]] = defaultdict(list)

    @contextmanager
    def transaction(self):
        """Re-entrant lock to guard expensive load/invalidation sequences."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def is_populated(self) -> bool:
        return self._loaded

    def data_view(self) -> Dict[str, List[Any]]:
        """Return the internal mapping for direct access (callers must mark dirty)."""
        return self._data

    def replace(self, mapping: Dict[str, List[Any]]) -> None:
        with self._lock:
            self._data = {key: list(values) for key, values in mapping.items()}
            self._loaded = True
            self._notify(mapping.keys())

    def set(self, key: str, values: List[Any]) -> None:
        with self._lock:
            self._data[key] = values
            self._loaded = True
            self._notify([key])

    def mark_dirty(self, kinds: Iterable[str]) -> None:
        with self._lock:
            self._notify(kinds)

    def register_callback(self, kind: str, callback: Callback) -> None:
        with self._lock:
            self._callbacks[kind].append(callback)

    def invalidate(self, kinds: Optional[Sequence[str]] = None) -> None:
        with self._lock:
            if kinds is None:
                removed_keys = list(self._data.keys())
                self._data = {}
                self._loaded = False
                self._notify(removed_keys)
                return

            removed: List[str] = []
            for kind in kinds:
                if kind in self._data:
                    removed.append(kind)
                    del self._data[kind]
            if removed:
                self._notify(removed)

    def clear(self) -> None:
        self.invalidate()

    def version(self) -> int:
        return self._version

    def _notify(self, kinds: Iterable[str]) -> None:
        unique: Set[str] = {kind for kind in kinds if kind}
        if not unique:
            return
        self._version += 1
        for kind in unique:
            for callback in self._callbacks.get(kind, []):
                try:
                    callback(kind)
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("ElementStore callback failed for '%s'", kind)
