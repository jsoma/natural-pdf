"""Delay native char TextElement construction until chars are requested."""

from contextlib import contextmanager

from experiments.performance import vector_metrics as metrics

METADATA = {
    "track": "vector",
    "candidate": "lazy_text_elements_upper_bound",
    "cache_only": False,
    "hypothesis": "Word-heavy flows can avoid eager char TextElement wrapping if raw char dicts are retained.",
}


@contextmanager
def install():
    from natural_pdf.core import element_manager as module
    from natural_pdf.core.element_manager import ElementManager
    from natural_pdf.elements.image import ImageElement
    from natural_pdf.elements.line import LineElement
    from natural_pdf.elements.rect import RectangleElement
    from natural_pdf.elements.text import TextElement

    original_populate_store = ElementManager._populate_store
    original_chars = ElementManager.chars.fget
    original_get_elements = ElementManager.get_elements

    def patched_populate_store(self):
        if self._load_text:
            native_chars = getattr(self._page._page, "chars", []) or []
            prepared_char_dicts = self._element_loader.prepare_native_chars(native_chars)
        else:
            prepared_char_dicts = []

        if self._load_text and prepared_char_dicts:
            self._decorations.annotate_chars(prepared_char_dicts)

        word_options = self._build_word_engine_options(prepared_char_dicts)
        generated_words = self._word_engine.generate_words(
            prepared_char_dicts,
            options=word_options,
            create_word_element=self._create_word_element,
            propagate_decorations=self._decorations.propagate_to_words,
            disable_text_sync=module.disable_text_sync,
        )

        self._npdf_perf_lazy_char_dicts = prepared_char_dicts
        elements_data = {
            "chars": [],
            "words": generated_words,
            "rects": [RectangleElement(rect, self._page) for rect in self._page._page.rects],
            "lines": [LineElement(line, self._page) for line in self._page._page.lines],
            "images": [ImageElement(image, self._page) for image in self._page._page.images],
        }

        if hasattr(self._page, "_regions") and (
            "detected" in self._page._regions
            or "named" in self._page._regions
            or "checkbox" in self._page._regions
        ):
            regions = []
            if "detected" in self._page._regions:
                regions.extend(self._page._regions["detected"])
            if "named" in self._page._regions:
                regions.extend(self._page._regions["named"].values())
            if "checkbox" in self._page._regions:
                regions.extend(self._page._regions["checkbox"])
            elements_data["regions"] = regions
        else:
            elements_data["regions"] = []

        metrics.count("vector.lazy_text_elements.deferred_chars", len(prepared_char_dicts))
        self._store.replace(elements_data)

    def materialize_chars(self):
        store = self._element_store()
        chars = list(store.get("chars", []))
        if chars:
            return chars
        char_dicts = getattr(self, "_npdf_perf_lazy_char_dicts", None)
        if not char_dicts:
            return original_chars(self)
        with metrics.timed("vector.lazy_text_elements.materialize_ms"):
            chars = [TextElement(char_dict, self._page) for char_dict in char_dicts]
        metrics.count("vector.lazy_text_elements.materialize_count")
        metrics.count("vector.lazy_text_elements.materialized_chars", len(chars))
        self._store.set("chars", chars)
        return list(chars)

    def patched_get_elements(self, element_type=None):
        if element_type == "chars":
            return materialize_chars(self)
        return original_get_elements(self, element_type)

    ElementManager._populate_store = patched_populate_store
    ElementManager.chars = property(materialize_chars)
    ElementManager.get_elements = patched_get_elements
    try:
        yield
    finally:
        ElementManager._populate_store = original_populate_store
        ElementManager.chars = property(original_chars)
        ElementManager.get_elements = original_get_elements
