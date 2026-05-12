"""Experiment-only metrics for vectorization prototypes."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

_COUNTS: Counter[str] = Counter()
_TIMINGS_MS: defaultdict[str, float] = defaultdict(float)
_BYTES: Counter[str] = Counter()


def reset() -> None:
    _COUNTS.clear()
    _TIMINGS_MS.clear()
    _BYTES.clear()


def count(label: str, amount: int | float = 1) -> None:
    _COUNTS[label] += amount


def timing(label: str, elapsed_ms: float) -> None:
    _TIMINGS_MS[label] += float(elapsed_ms)


def byte_count(label: str, amount: int | float) -> None:
    _BYTES[label] += amount


def array_bytes(label: str, *arrays: Any) -> None:
    total = 0
    for array in arrays:
        total += int(getattr(array, "nbytes", 0) or 0)
    if total:
        byte_count(label, total)


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timing(label, (time.perf_counter() - start) * 1000.0)


def snapshot() -> dict[str, dict[str, float]]:
    return {
        "counts": dict(_COUNTS),
        "timings_ms": dict(_TIMINGS_MS),
        "bytes": dict(_BYTES),
    }
