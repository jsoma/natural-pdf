from __future__ import annotations

from typing import Any, Iterable


def _collect_unique_pages(host: Any, elements: Iterable[Any] = ()) -> list[Any]:
    pages: list[Any] = []
    seen_ids: set[int] = set()

    def _add_page(candidate: Any) -> None:
        if candidate is None:
            return
        marker = id(candidate)
        if marker in seen_ids:
            return
        seen_ids.add(marker)
        pages.append(candidate)

    host_page = host if hasattr(host, "_bump_text_state_version") else getattr(host, "page", None)
    _add_page(host_page)

    for element in elements:
        _add_page(getattr(element, "page", None))

    return pages


def bump_text_state(host: Any, *, elements: Iterable[Any] = ()) -> None:
    for page in _collect_unique_pages(host, elements):
        bump = getattr(page, "_bump_text_state_version", None)
        if callable(bump):
            bump()
