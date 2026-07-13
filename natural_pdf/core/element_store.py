from __future__ import annotations

import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
        with self._lock:
            return self._loaded

    def data_view(self) -> Dict[str, List[Any]]:
        """Return the internal mapping for legacy manager internals.

        New code should use :meth:`snapshot`, :meth:`append_unique`,
        :meth:`upsert`, and :meth:`remove_where` so mutations remain atomic.
        Callers of this compatibility escape hatch must hold ``transaction()``
        while mutating and call :meth:`mark_dirty` afterward.
        """
        return self._data

    def snapshot(self, key: str) -> Tuple[Any, ...]:
        """Return an immutable point-in-time snapshot of one element kind."""

        with self._lock:
            return tuple(self._data.get(key, ()))

    def mapping_snapshot(self) -> Dict[str, Tuple[Any, ...]]:
        """Return an immutable-values snapshot of all element kinds."""

        with self._lock:
            return {key: tuple(values) for key, values in self._data.items()}

    def replace(self, mapping: Dict[str, List[Any]]) -> None:
        with self._lock:
            self._data = {key: list(values) for key, values in mapping.items()}
            self._loaded = True
            self._notify(mapping.keys())

    def set(self, key: str, values: List[Any]) -> None:
        with self._lock:
            # Never retain a caller-owned mutable list.  Otherwise a later
            # append/remove can bypass versioning and invalidation callbacks.
            self._data[key] = list(values)
            self._loaded = True
            self._notify([key])

    def append_unique(self, key: str, item: Any) -> bool:
        """Append ``item`` once by identity and notify atomically."""

        with self._lock:
            current = self._data.setdefault(key, [])
            if any(existing is item for existing in current):
                return False
            current.append(item)
            self._loaded = True
            self._notify([key])
            return True

    def upsert(
        self,
        key: str,
        item: Any,
        *,
        replace_if: Callable[[Any], bool],
    ) -> Tuple[bool, Tuple[Any, ...]]:
        """Atomically replace matching entries with ``item``.

        The replacement occupies the first matching entry's position and any
        additional matches are removed.  If there is no match, ``item`` is
        appended.  This preserves stable iteration order for keyed repositories
        such as named Regions.

        Returns:
            ``(changed, replaced_items)``.  Re-registering the exact sole item
            is a no-op.
        """

        with self._lock:
            current = self._data.setdefault(key, [])
            matches = [existing for existing in current if replace_if(existing)]
            if len(matches) == 1 and matches[0] is item:
                return False, ()

            replacement: List[Any] = []
            inserted = False
            for existing in current:
                if replace_if(existing):
                    if not inserted:
                        replacement.append(item)
                        inserted = True
                    continue
                # Maintain the store-wide identity uniqueness invariant even
                # if a caller re-registers an item under a different key.
                if existing is item:
                    continue
                replacement.append(existing)
            if not inserted:
                replacement.append(item)

            self._data[key] = replacement
            self._loaded = True
            self._notify([key])
            return True, tuple(existing for existing in matches if existing is not item)

    def remove_where(self, key: str, predicate: Callable[[Any], bool]) -> Tuple[Any, ...]:
        """Atomically remove and return entries matching ``predicate``."""

        with self._lock:
            current = self._data.get(key, [])
            removed = tuple(item for item in current if predicate(item))
            if not removed:
                return ()
            removed_ids = {id(item) for item in removed}
            self._data[key] = [item for item in current if id(item) not in removed_ids]
            self._notify([key])
            return removed

    def mark_dirty(self, kinds: Iterable[str]) -> None:
        with self._lock:
            self._notify(kinds)

    def register_callback(self, kind: str, callback: Callback) -> None:
        with self._lock:
            self._callbacks[kind].append(callback)

    def invalidate(
        self,
        kinds: Optional[Sequence[str]] = None,
        *,
        preserve_kinds: Sequence[str] = (),
    ) -> None:
        with self._lock:
            if kinds is None:
                preserved = {
                    key: list(self._data[key]) for key in preserve_kinds if key in self._data
                }
                removed_keys = [key for key in self._data if key not in preserved]
                self._data = preserved
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

    def clear(self, *, preserve_kinds: Sequence[str] = ()) -> None:
        self.invalidate(preserve_kinds=preserve_kinds)

    def version(self) -> int:
        with self._lock:
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
