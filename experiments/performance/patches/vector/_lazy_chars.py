"""Shared lazy native-char materialization helpers for experiments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable

from experiments.performance import vector_metrics as metrics

_LAZY_DICT_ATTRS = (
    "_npdf_perf_lazy_char_dicts",
    "_npdf_perf_raw_char_dicts",
    "_npdf_perf_columnar_char_dicts",
)
_LAZY_PROXY_ATTR = "_npdf_perf_lazy_char_proxy"
_LAZY_ARRAY_ATTR = "_npdf_perf_columnar_char_array"


class LazyCharElementList(Sequence[Any]):
    """List-like char collection that wraps raw dicts only when iterated/indexed."""

    def __init__(self, manager: Any, char_dicts: list[dict[str, Any]], *, label: str) -> None:
        self._manager = manager
        self._char_dicts = char_dicts
        self._label = label
        self._materialized: list[Any] | None = None

    def _materialize(self) -> list[Any]:
        if self._materialized is None:
            from natural_pdf.elements.text import TextElement

            with metrics.timed(f"{self._label}.materialize_ms"):
                self._materialized = [
                    TextElement(char_dict, self._manager._page) for char_dict in self._char_dicts
                ]
            metrics.count(f"{self._label}.materialize_count")
            metrics.count(f"{self._label}.materialized_chars", len(self._materialized))
            self._manager._store.set("chars", self._materialized)
        return self._materialized

    def __len__(self) -> int:
        metrics.count(f"{self._label}.proxy_len_calls")
        if self._materialized is not None:
            return len(self._materialized)
        return len(self._char_dicts)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._materialize())

    def __getitem__(self, index):
        return self._materialize()[index]

    def __contains__(self, item: object) -> bool:
        return item in self._materialize()

    def append(self, item: Any) -> None:
        self._materialize().append(item)

    def extend(self, items) -> None:
        self._materialize().extend(items)

    def remove(self, item: Any) -> None:
        self._materialize().remove(item)


def _build_regions(manager: Any) -> list[Any]:
    page = manager._page
    if not hasattr(page, "_regions") or not (
        "detected" in page._regions or "named" in page._regions or "checkbox" in page._regions
    ):
        return []

    regions = []
    if "detected" in page._regions:
        regions.extend(page._regions["detected"])
    if "named" in page._regions:
        regions.extend(page._regions["named"].values())
    if "checkbox" in page._regions:
        regions.extend(page._regions["checkbox"])
    return regions


def _clear_lazy_state(manager: Any) -> None:
    for attr in (*_LAZY_DICT_ATTRS, _LAZY_PROXY_ATTR, _LAZY_ARRAY_ATTR):
        if hasattr(manager, attr):
            delattr(manager, attr)


def _raw_char_dicts(manager: Any) -> list[dict[str, Any]]:
    for attr in _LAZY_DICT_ATTRS:
        value = getattr(manager, attr, None)
        if value is not None:
            return value
    proxy = getattr(manager, _LAZY_PROXY_ATTR, None)
    if proxy is not None:
        return proxy._char_dicts
    return []


def _store_chars(manager: Any) -> Any:
    store = manager._element_store()
    return store.get("chars", [])


def materialize_chars(manager: Any, *, label: str, original_chars: Callable[[Any], list[Any]]):
    store = manager._element_store()
    chars = store.get("chars", [])
    if isinstance(chars, LazyCharElementList):
        return list(chars)
    if chars:
        return list(chars)

    char_dicts = _raw_char_dicts(manager)
    if not char_dicts:
        return original_chars(manager)

    from natural_pdf.elements.text import TextElement

    with metrics.timed(f"{label}.materialize_ms"):
        materialized = [TextElement(char_dict, manager._page) for char_dict in char_dicts]
    metrics.count(f"{label}.materialize_count")
    metrics.count(f"{label}.materialized_chars", len(materialized))
    manager._store.set("chars", materialized)
    return list(materialized)


def populate_lazy_chars(manager: Any, *, mode: str, label: str) -> None:
    from experiments.performance.vector_helpers import element_array
    from natural_pdf.core import element_manager as module
    from natural_pdf.elements.image import ImageElement
    from natural_pdf.elements.line import LineElement
    from natural_pdf.elements.rect import RectangleElement

    _clear_lazy_state(manager)

    if manager._load_text:
        native_chars = getattr(manager._page._page, "chars", []) or []
        prepared_char_dicts = manager._element_loader.prepare_native_chars(native_chars)
    else:
        prepared_char_dicts = []

    if manager._load_text and prepared_char_dicts:
        manager._decorations.annotate_chars(prepared_char_dicts)

    word_options = manager._build_word_engine_options(prepared_char_dicts)
    generated_words = manager._word_engine.generate_words(
        prepared_char_dicts,
        options=word_options,
        create_word_element=manager._create_word_element,
        propagate_decorations=manager._decorations.propagate_to_words,
        disable_text_sync=module.disable_text_sync,
    )

    proxy = None
    chars_collection: list[Any] = []
    if mode == "minimal":
        manager._npdf_perf_lazy_char_dicts = prepared_char_dicts
    elif mode == "proxy":
        proxy = LazyCharElementList(manager, prepared_char_dicts, label=label)
        manager._npdf_perf_lazy_char_proxy = proxy
    elif mode == "raw_store":
        manager._npdf_perf_raw_char_dicts = prepared_char_dicts
    elif mode == "columnar_store":
        manager._npdf_perf_columnar_char_dicts = prepared_char_dicts
        manager._npdf_perf_columnar_char_array = element_array(prepared_char_dicts, label=label)
    else:  # pragma: no cover - patch authoring guard
        raise ValueError(f"Unknown lazy char mode: {mode}")

    elements_data = {
        "chars": chars_collection,
        "words": generated_words,
        "rects": [RectangleElement(rect, manager._page) for rect in manager._page._page.rects],
        "lines": [LineElement(line, manager._page) for line in manager._page._page.lines],
        "images": [ImageElement(image, manager._page) for image in manager._page._page.images],
        "regions": _build_regions(manager),
    }
    metrics.count(f"{label}.deferred_chars", len(prepared_char_dicts))
    manager._store.replace(elements_data)

    if proxy is not None:
        # ElementStore.replace() intentionally list-copies collections, so install
        # the proxy after replace() to keep this variant lazy.
        manager._store.data_view()["chars"] = proxy


@contextmanager
def install_lazy_chars_variant(*, mode: str, label: str):
    from natural_pdf.core.element_manager import ElementManager

    original_populate_store = ElementManager._populate_store
    original_chars = ElementManager.chars.fget
    original_get_elements = ElementManager.get_elements
    original_get_all_elements = ElementManager.get_all_elements
    original_add_element = ElementManager.add_element
    original_remove_element = ElementManager.remove_element
    original_remove_elements_by_source = ElementManager.remove_elements_by_source
    original_remove_ocr_elements = ElementManager.remove_ocr_elements
    original_clear_text_layer = ElementManager.clear_text_layer
    original_invalidate_cache = ElementManager.invalidate_cache

    def patched_populate_store(self):
        return populate_lazy_chars(self, mode=mode, label=label)

    def patched_chars(self):
        return materialize_chars(self, label=label, original_chars=original_chars)

    def patched_get_elements(self, element_type=None):
        if element_type in ("chars", "char"):
            return materialize_chars(self, label=label, original_chars=original_chars)
        if element_type is None:
            return patched_get_all_elements(self)
        return original_get_elements(self, element_type)

    def patched_get_all_elements(self):
        try:
            store = self._element_store()
        except RuntimeError:
            return []
        all_elements = []
        for key, elements in store.items():
            if key == "chars":
                all_elements.extend(
                    materialize_chars(self, label=label, original_chars=original_chars)
                )
            else:
                all_elements.extend(elements)
        return all_elements

    def patched_add_element(self, element, element_type="words"):
        if element_type in ("chars", "char"):
            materialize_chars(self, label=label, original_chars=original_chars)
            element_type = "chars"
        return original_add_element(self, element, element_type)

    def patched_remove_element(self, element, element_type="words"):
        if element_type in ("chars", "char"):
            materialize_chars(self, label=label, original_chars=original_chars)
            element_type = "chars"
        return original_remove_element(self, element, element_type)

    def patched_remove_elements_by_source(self, element_type: str, source: str) -> int:
        if element_type in ("chars", "char"):
            materialize_chars(self, label=label, original_chars=original_chars)
            element_type = "chars"
        return original_remove_elements_by_source(self, element_type, source)

    def patched_remove_ocr_elements(self):
        if _raw_char_dicts(self) or isinstance(_store_chars(self), LazyCharElementList):
            materialize_chars(self, label=label, original_chars=original_chars)
        return original_remove_ocr_elements(self)

    def patched_clear_text_layer(self):
        raw_chars = _raw_char_dicts(self)
        store = self._element_store()
        store_chars = store.get("chars", [])
        removed_chars = len(raw_chars) if raw_chars and not store_chars else len(store_chars)
        removed_words = len(store.get("words", []))
        _clear_lazy_state(self)
        if "words" in store:
            self._store.set("words", [])
        if "chars" in store:
            self._store.set("chars", [])
        return removed_words, removed_chars

    def patched_invalidate_cache(self):
        _clear_lazy_state(self)
        return original_invalidate_cache(self)

    ElementManager._populate_store = patched_populate_store
    ElementManager.chars = property(patched_chars)
    ElementManager.get_elements = patched_get_elements
    ElementManager.get_all_elements = patched_get_all_elements
    ElementManager.add_element = patched_add_element
    ElementManager.remove_element = patched_remove_element
    ElementManager.remove_elements_by_source = patched_remove_elements_by_source
    ElementManager.remove_ocr_elements = patched_remove_ocr_elements
    ElementManager.clear_text_layer = patched_clear_text_layer
    ElementManager.invalidate_cache = patched_invalidate_cache
    try:
        yield
    finally:
        ElementManager._populate_store = original_populate_store
        ElementManager.chars = property(original_chars)
        ElementManager.get_elements = original_get_elements
        ElementManager.get_all_elements = original_get_all_elements
        ElementManager.add_element = original_add_element
        ElementManager.remove_element = original_remove_element
        ElementManager.remove_elements_by_source = original_remove_elements_by_source
        ElementManager.remove_ocr_elements = original_remove_ocr_elements
        ElementManager.clear_text_layer = original_clear_text_layer
        ElementManager.invalidate_cache = original_invalidate_cache
