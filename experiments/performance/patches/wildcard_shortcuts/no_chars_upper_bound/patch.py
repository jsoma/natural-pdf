"""Upper bound: omit native chars from broad all-element pools."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.performance import vector_metrics as metrics

METADATA = {
    "track": "wildcard_shortcuts",
    "candidate": "no_chars_upper_bound",
    "cache_only": False,
    "unsafe_semantics": True,
    "hypothesis": "If broad any-element paths do not need char elements, avoiding char materialization should reduce accidental one-shot costs.",
}


@contextmanager
def install():
    from natural_pdf.core.element_manager import ElementManager

    original_get_all_elements = ElementManager.get_all_elements

    def patched_get_all_elements(self):
        try:
            store = self._element_store()
        except RuntimeError:
            return []

        all_elements = []
        for element_type, elements in store.items():
            if element_type == "chars":
                metrics.count("shortcut.wildcard_no_chars.skipped_chars", len(elements))
                raw_chars = getattr(self, "_raw_char_dicts", None)
                if raw_chars is not None:
                    metrics.count("shortcut.wildcard_no_chars.skipped_raw_chars", len(raw_chars))
                continue
            all_elements.extend(elements)
        metrics.count("shortcut.wildcard_no_chars.fast_path")
        return all_elements

    ElementManager.get_all_elements = patched_get_all_elements
    try:
        yield
    finally:
        ElementManager.get_all_elements = original_get_all_elements
