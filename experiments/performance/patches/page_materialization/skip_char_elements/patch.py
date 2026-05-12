"""Skip eager char TextElement wrapping while preserving word generation."""

from contextlib import contextmanager

METADATA = {
    "track": "page_materialization",
    "candidate": "skip_char_elements",
    "cache_only": False,
    "hypothesis": "Eager native-char TextElement wrapping is a measurable page materialization cost.",
}


@contextmanager
def install():
    from natural_pdf.core import element_manager as module
    from natural_pdf.core.element_manager import ElementManager
    from natural_pdf.elements.image import ImageElement
    from natural_pdf.elements.line import LineElement
    from natural_pdf.elements.rect import RectangleElement

    original_populate_store = ElementManager._populate_store

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

        elements_data = {
            "chars": [],
            "words": generated_words,
            "rects": [RectangleElement(r, self._page) for r in self._page._page.rects],
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

        self._store.replace(elements_data)

    ElementManager._populate_store = patched_populate_store
    try:
        yield
    finally:
        ElementManager._populate_store = original_populate_store
