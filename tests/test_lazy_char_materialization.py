from natural_pdf import PDF


def test_extract_text_does_not_materialize_native_char_elements():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]

        text = page.extract_text()

        assert "Jungle Health and Safety" in text
        store = page._element_mgr._store.data_view()
        assert len(store["words"]) > 0
        assert store["chars"] == []
        assert page._element_mgr._raw_char_dicts
    finally:
        pdf.close()


def test_page_chars_materializes_deferred_native_chars():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        native_char_count = len(page._page.chars)

        assert len(page.words) > 0
        assert page._element_mgr._store.data_view()["chars"] == []

        chars = page.chars

        assert len(chars) == native_char_count
        assert len(page._element_mgr._store.data_view()["chars"]) == native_char_count
        assert page._element_mgr._raw_char_dicts is None
    finally:
        pdf.close()


def test_find_all_char_materializes_and_preserves_char_results():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        native_char_count = len(page._page.chars)

        chars = page.find_all("char")

        assert len(chars) == native_char_count
        assert chars[0].text == page._page.chars[0]["text"]
        assert page._element_mgr._raw_char_dicts is None
    finally:
        pdf.close()


def test_clear_text_layer_counts_deferred_native_chars():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        native_char_count = len(page._page.chars)
        word_count = len(page.words)

        removed_words, removed_chars = page._element_mgr.clear_text_layer()

        assert removed_words == word_count
        assert removed_chars == native_char_count
        assert page.words == []
        assert page.chars == []
    finally:
        pdf.close()
