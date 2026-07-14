import threading

import pytest

from natural_pdf import PDF
from natural_pdf.core.element_manager import disable_text_sync
from natural_pdf.core.word_engine import WordEngine, WordEngineOptions
from natural_pdf.elements.text import TextElement
from natural_pdf.flows.flow import Flow


@pytest.fixture
def mutable_pdf():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        yield pdf
    finally:
        pdf.close()


def _indexed_word_pair(page):
    words = page.words
    for index, word in enumerate(words[:-1]):
        following = words[index + 1]
        if (
            len(word.text) >= 6
            and len(word._char_indices) == len(word.text)
            and following._char_indices
        ):
            return word, following
    raise AssertionError("Expected two adjacent indexed words in the practice PDF")


def _char_dict(text, x0, *, source="manual-test"):
    return {
        "text": text,
        "x0": float(x0),
        "top": 10.0,
        "x1": float(x0 + 5),
        "bottom": 20.0,
        "doctop": 10.0,
        "y0": 772.0,
        "y1": 782.0,
        "width": 5.0,
        "height": 10.0,
        "adv": 5.0,
        "fontname": "TestFont",
        "size": 10.0,
        "upright": True,
        "bold": False,
        "italic": False,
        "non_stroking_color": None,
        "strike": False,
        "underline": False,
        "highlight": False,
        "highlight_color": None,
        "object_type": "char",
        "source": source,
    }


def _manual_word(page, text, char_dicts, *, source="manual-test"):
    word = TextElement(
        {
            "text": text,
            "x0": min(char["x0"] for char in char_dicts),
            "top": min(char["top"] for char in char_dicts),
            "x1": max(char["x1"] for char in char_dicts),
            "bottom": max(char["bottom"] for char in char_dicts),
            "object_type": "word",
            "source": source,
            "_char_dicts": char_dicts,
        },
        page,
    )
    word._text_manually_set = True
    assert page.add_element(word, "words")
    return word


def _install_synthetic_native_chars(page, monkeypatch, chars, **config):
    manager = page._element_mgr
    manager.invalidate_cache(preserve_overlays=False)
    page._config.update(
        {
            "auto_text_tolerance": False,
            "keep_blank_chars": True,
            "space_gap_ratio": 0,
            **config,
        }
    )
    monkeypatch.setattr(
        manager._element_loader,
        "prepare_native_chars",
        lambda _native: [char.copy() for char in chars],
    )
    return manager


def test_word_engine_indexes_coincident_glyphs_by_dict_identity(mutable_pdf):
    page = mutable_pdf.pages[0]
    chars = [_char_dict("A", 10, source="native") for _ in range(2)]
    # Make the glyphs truly coincident, including their full geometry and text.
    chars[1].update(chars[0])
    engine = WordEngine([], load_text=True)
    options = WordEngineOptions(
        page_number=page.number,
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=True,
        x_tolerance_ratio=None,
        y_tolerance_ratio=None,
        space_gap_ratio=0,
    )

    words = engine.generate_words(
        chars,
        options=options,
        create_word_element=page._element_mgr._create_word_element,
        propagate_decorations=lambda _words, _chars: None,
        disable_text_sync=disable_text_sync,
    )

    assert sorted(index for word in words for index in word._char_indices) == [0, 1]


def test_direct_word_edits_reindex_following_word_for_shorter_and_longer_text(mutable_pdf):
    page = mutable_pdf.pages[0]
    edited, following = _indexed_word_pair(page)
    original_text = edited.text
    original_occurrences = page.extract_text().count(original_text)
    following_text = following.text
    following_chars = list(following.chars)
    initial_version = page._text_state_version

    edited.text = "ZXQ"

    assert edited.text == "ZXQ"
    assert "".join(char.text for char in edited.chars) == "ZXQ"
    assert all(char in page.find_all("char") for char in edited.chars)
    assert following.text == following_text
    assert following.chars == following_chars
    assert "".join(char.text for char in following.chars) == following_text
    assert edited in page.find_all('text:contains("ZXQ")')
    assert page.extract_text().count(original_text) == original_occurrences - 1
    assert page._text_state_version == initial_version + 1

    longer_text = "REVISION-LONGER-THAN-BEFORE"
    edited.text = longer_text

    assert edited.text == longer_text
    assert "".join(char.text for char in edited.chars) == longer_text
    assert following.text == following_text
    assert following.chars == following_chars
    assert "".join(char.text for char in following.chars) == following_text
    assert longer_text in page.extract_text()
    assert "ZXQ" not in page.extract_text()
    assert page._text_state_version == initial_version + 2

    equal_length_text = "E" * len(longer_text)
    edited.text = equal_length_text
    assert "".join(char.text for char in edited.chars) == equal_length_text
    assert following.text == following_text
    assert following.chars == following_chars


def test_text_mutation_invalidates_cached_style_summary(mutable_pdf):
    page = mutable_pdf.pages[0]
    word, _ = _indexed_word_pair(page)
    page._text_styles_summary = {"old-style": {"label": "Old"}}
    page._text_styles = object()
    page.metadata["text_styles_summary"] = {"old-style": {"label": "Old"}}

    word.text = f"{word.text}-EDITED"

    assert page._text_styles_summary == {}
    assert page._text_styles is None
    assert "text_styles_summary" not in page.metadata


def test_word_edit_copies_chars_shared_with_another_word(mutable_pdf):
    page = mutable_pdf.pages[0]
    edited, _ = _indexed_word_pair(page)
    original_text = edited.text
    original_char_dicts = list(edited._char_dicts)
    alias = _manual_word(page, original_text, original_char_dicts, source="alias-test")

    edited.text = "COPIED"

    assert edited.text == "COPIED"
    assert "".join(char.text for char in edited.chars) == "COPIED"
    assert alias.text == original_text
    assert "".join(char.text for char in alias.chars) == original_text
    assert alias._char_dicts == original_char_dicts
    assert all(
        edited_dict is not alias_dict
        for edited_dict, alias_dict in zip(edited._char_dicts, alias._char_dicts)
    )


def test_direct_char_edit_updates_word_extraction_and_selectors(mutable_pdf):
    page = mutable_pdf.pages[0]
    word = next(
        item
        for item in page.words
        if len(item.text) >= 4 and len(item._char_indices) == len(item.text)
    )
    char = word.chars[0]
    replacement = "Q" if char.text != "Q" else "Z"
    expected = replacement + word.text[1:]

    char.text = replacement

    assert word.text == expected
    assert word.extract_text() == expected
    assert word in page.find_all(f'text:contains("{expected}")')
    assert expected in page.extract_text()


def test_direct_char_edit_keeps_hidden_boundary_whitespace_hidden(mutable_pdf):
    page = mutable_pdf.pages[0]
    chars = [
        _char_dict(" ", 10),
        _char_dict("A", 15),
        _char_dict("B", 20),
        _char_dict(" ", 25),
    ]
    word = _manual_word(page, "AB", chars)

    word.chars[1].text = "Q"

    assert word.text == "QB"
    assert word.extract_text() == "QB"
    assert "".join(char.text for char in word.chars) == " QB "


def test_direct_char_edit_preserves_word_engine_inferred_space(mutable_pdf):
    page = mutable_pdf.pages[0]
    char_dicts = [
        {
            "text": "A",
            "x0": 10,
            "top": 10,
            "x1": 15,
            "bottom": 20,
            "object_type": "char",
            "source": "manual-test",
        },
        {
            "text": "B",
            "x0": 25,
            "top": 10,
            "x1": 30,
            "bottom": 20,
            "object_type": "char",
            "source": "manual-test",
        },
    ]
    word = TextElement(
        {
            "text": "A B",
            "x0": 10,
            "top": 10,
            "x1": 30,
            "bottom": 20,
            "object_type": "word",
            "source": "manual-test",
            "_char_dicts": char_dicts,
        },
        page,
    )
    word._text_manually_set = True
    assert page.add_element(word, "words")

    word.chars[1].text = "C"

    assert word.text == "A C"
    assert word.extract_text() == "A C"


def test_word_edit_preserves_unexposed_trailing_layout_space(mutable_pdf):
    page = mutable_pdf.pages[0]
    words = page.words
    index = next(index for index, word in enumerate(words) if word.text == "Site:")
    word = words[index]
    following = words[index + 1]
    assert "".join(char.text for char in word.chars).endswith(" ")

    word.text = "EDITED-SITE"

    assert word.text == "EDITED-SITE"
    assert "".join(char.text for char in word.chars) == "EDITED-SITE "
    assert f"EDITED-SITE {following.text}" in page.extract_text()


def test_detached_text_edit_does_not_publish_page_mutation(mutable_pdf):
    page = mutable_pdf.pages[0]
    detached = TextElement(
        {
            "text": "DETACHED",
            "x0": 10,
            "top": 10,
            "x1": 70,
            "bottom": 20,
            "object_type": "word",
            "source": "ocr",
            "_char_dicts": [
                {
                    "text": "DETACHED",
                    "x0": 10,
                    "top": 10,
                    "x1": 70,
                    "bottom": 20,
                    "object_type": "char",
                    "source": "ocr",
                }
            ],
        },
        page,
    )
    version = page._text_state_version

    detached.text = "LOCAL"

    assert detached.text == "LOCAL"
    assert page._text_state_version == version


def test_detached_text_edit_copies_shared_canonical_char_dicts(mutable_pdf):
    page = mutable_pdf.pages[0]
    canonical, _ = _indexed_word_pair(page)
    original_text = canonical.text
    original_dicts = list(canonical._char_dicts)
    detached = TextElement(
        {
            **canonical._obj,
            "text": original_text,
            "object_type": "word",
            "_char_dicts": original_dicts,
        },
        page,
    )

    detached.text = "Z" * len(original_text)

    assert canonical.text == original_text
    assert "".join(char.text for char in canonical.chars) == original_text
    assert all(
        detached_dict is not canonical_dict
        for detached_dict, canonical_dict in zip(detached._char_dicts, original_dicts)
    )


def test_text_sync_suppression_is_thread_local(mutable_pdf):
    page = mutable_pdf.pages[0]
    suppressed_word, edited_word = _indexed_word_pair(page)
    suppressed_original = suppressed_word.text
    edited_value = "THREADSAFE"
    suppression_started = threading.Event()
    edit_finished = threading.Event()

    def hold_suppression_open():
        with disable_text_sync():
            suppression_started.set()
            assert edit_finished.wait(timeout=5)
            # Restore before leaving: this path models construction-only use.
            suppressed_word.text = "TEMPORARY"
            suppressed_word.text = suppressed_original

    worker = threading.Thread(target=hold_suppression_open)
    worker.start()
    assert suppression_started.wait(timeout=5)
    edited_word.text = edited_value
    edit_finished.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert suppressed_word.text == suppressed_original
    assert "".join(char.text for char in suppressed_word.chars) == suppressed_original
    assert edited_word.text == edited_value
    assert "".join(char.text for char in edited_word.chars) == edited_value


def test_flow_recomputes_text_and_elements_after_mutations_and_exclusions(mutable_pdf):
    page = mutable_pdf.pages[0]
    flow = Flow([page], arrangement="vertical")
    flow_region = flow._analysis_region()
    title = page.words[0]
    original_title = title.text

    first_elements = flow_region.elements()
    second_elements = flow_region.elements()
    assert first_elements is not second_elements
    initial_text = flow_region.extract_text()
    assert original_title in initial_text

    title.text = "FLOW-MUTATION"
    mutated_text = flow_region.extract_text()
    assert "FLOW-MUTATION" in mutated_text
    assert mutated_text.count(original_title) == initial_text.count(original_title) - 1

    page.add_exclusion(title)
    assert title not in flow_region.elements()
    assert "FLOW-MUTATION" not in flow_region.extract_text()

    page.clear_exclusions()
    assert title in flow_region.elements()
    assert "FLOW-MUTATION" in flow_region.extract_text()

    ocr_word = page._element_mgr.create_text_elements_from_ocr(
        [{"text": "FLOW-OCR-ADDITION", "bbox": (10, 10, 100, 25), "confidence": 0.9}],
        engine_name="test",
    )[0]
    assert ocr_word in flow_region.elements()
    assert "FLOW-OCR-ADDITION" in flow_region.extract_text()

    page._element_mgr.remove_ocr_elements()
    assert ocr_word not in flow_region.elements()
    assert "FLOW-OCR-ADDITION" not in flow_region.extract_text()


def test_flow_recomputes_for_pdf_exclusions_and_extract_kwargs(mutable_pdf):
    page = mutable_pdf.pages[0]
    flow_region = Flow([page], arrangement="vertical")._analysis_region()
    title = page.words[0]

    with_pipes = flow_region.extract_text(newlines="|")
    flattened = flow_region.extract_text(newlines=False)
    assert with_pipes != flattened
    assert "|" in with_pipes
    assert "|" not in flattened

    title_occurrences = flow_region.extract_text().count(title.text)
    mutable_pdf.add_exclusion(lambda current: current.region(*title.bbox))
    assert flow_region.extract_text().count(title.text) == title_occurrences - 1

    mutable_pdf.clear_exclusions()
    assert flow_region.extract_text().count(title.text) == title_occurrences


def test_native_reload_preserves_ocr_overlay_and_manual_edit(mutable_pdf):
    page = mutable_pdf.pages[0]
    manager = page._element_mgr
    overlay = manager.create_text_elements_from_ocr(
        [{"text": "OVERLAY-TOKEN", "bbox": (10, 10, 90, 25), "confidence": 0.99}],
        engine_name="test",
    )[0]
    native_word = page.words[0]
    native_word.text = "EDITED-NATIVE"

    with page._temporary_text_settings({"x_tolerance": 9}):
        assert overlay in page.words
        assert native_word in page.words
        assert "OVERLAY-TOKEN" in page.extract_text()
        assert "EDITED-NATIVE" in page.extract_text()

    assert overlay in page.words
    assert native_word in page.words
    assert "OVERLAY-TOKEN" in page.extract_text()
    assert "EDITED-NATIVE" in page.extract_text()


def test_inferred_space_is_recomputed_instead_of_pinned_on_reload(mutable_pdf, monkeypatch):
    page = mutable_pdf.pages[0]
    chars = [_char_dict("A", 10, source="native"), _char_dict("B", 17, source="native")]
    manager = _install_synthetic_native_chars(
        page,
        monkeypatch,
        chars,
        x_tolerance=3,
        y_tolerance=3,
        space_gap_ratio=0.12,
    )

    injected = page.words
    assert [word.text for word in injected] == ["A B"]
    assert injected[0]._text_manually_set
    assert not injected[0]._text_user_edited

    page._config["space_gap_ratio"] = 0
    manager.invalidate_cache()

    assert [word.text for word in page.words] == ["AB"]
    assert injected[0] not in page.words


def test_reload_filters_edited_glyphs_before_regrouping(mutable_pdf, monkeypatch):
    page = mutable_pdf.pages[0]
    chars = [_char_dict("A", 10, source="native"), _char_dict("B", 16, source="native")]
    manager = _install_synthetic_native_chars(
        page,
        monkeypatch,
        chars,
        x_tolerance=0,
        y_tolerance=3,
    )
    initial_words = page.words
    assert [word.text for word in initial_words] == ["A", "B"]
    edited = initial_words[0]
    edited.text = "X"

    # Without pre-filtering, the fresh "AB" group overlaps the edited A origin
    # and is discarded wholesale, taking the untouched B glyph with it.
    page._config["x_tolerance"] = 3
    manager.invalidate_cache()

    assert sorted(word.text for word in page.words) == ["B", "X"]
    assert edited in page.words


def test_remove_word_keeps_shared_chars_and_removes_exclusive_chars(mutable_pdf):
    page = mutable_pdf.pages[0]
    shared = _char_dict("S", 10, source="shared-char")
    exclusive = _char_dict("X", 15, source="remove-char")
    removed_word = _manual_word(page, "SX", [shared, exclusive], source="remove-word")
    retained_word = _manual_word(page, "S", [shared], source="keep-word")

    assert page.remove_element(removed_word)

    remaining_chars = page.find_all("char")
    assert retained_word in page.words
    assert retained_word.chars[0]._obj is shared
    assert any(char._obj is shared for char in remaining_chars)
    assert not any(char._obj is exclusive for char in remaining_chars)


def test_remove_words_by_source_repairs_shared_char_references(mutable_pdf):
    page = mutable_pdf.pages[0]
    shared = _char_dict("S", 10, source="shared-char")
    exclusive = _char_dict("X", 15, source="exclusive-char")
    _manual_word(page, "SX", [shared, exclusive], source="remove-source")
    retained_word = _manual_word(page, "S", [shared], source="keep-source")

    assert page.remove_elements_by_source("words", "remove-source") == 1

    remaining_chars = page.find_all("char")
    assert retained_word in page.words
    assert "".join(char.text for char in retained_word.chars) == "S"
    assert any(char._obj is shared for char in remaining_chars)
    assert not any(char._obj is exclusive for char in remaining_chars)


def test_remove_char_updates_owning_word_without_dangling_index(mutable_pdf):
    page = mutable_pdf.pages[0]
    word, _ = _indexed_word_pair(page)
    original_text = word.text
    removed_char = word.chars[1]

    assert page.remove_element(removed_char)

    assert removed_char not in page.find_all("char")
    assert word.text == original_text[0] + original_text[2:]
    assert "".join(char.text for char in word.chars) == word.text
    assert all(index < len(page.find_all("char")) for index in word._char_indices)


def test_coincident_native_glyph_occurrences_are_preserved_one_to_one(mutable_pdf):
    page = mutable_pdf.pages[0]
    manager = page._element_mgr
    base_char = {
        "text": "A",
        "x0": 10.0,
        "top": 20.0,
        "x1": 16.0,
        "bottom": 30.0,
        "fontname": "StackedFont",
        "size": 10.0,
        "source": "native",
        "object_type": "char",
    }
    old_chars = [base_char.copy(), base_char.copy()]
    reloaded_chars = [base_char.copy(), base_char.copy()]
    manager._assign_native_origin_keys(old_chars)
    manager._assign_native_origin_keys(reloaded_chars)

    assert manager._char_origin_key(old_chars[0]) != manager._char_origin_key(old_chars[1])
    assert [manager._char_origin_key(item) for item in old_chars] == [
        manager._char_origin_key(item) for item in reloaded_chars
    ]
    old_chars[1]["text"] = "EDITED"

    def make_word(char_dict, text="A"):
        word = TextElement(
            {
                **base_char,
                "text": text,
                "object_type": "word",
                "_char_dicts": [char_dict],
            },
            page,
        )
        word._native_origin_keys = (manager._char_origin_key(char_dict),)
        return word

    preserved_word = make_word(old_chars[1], text="EDITED")
    preserved_word._set_text_value("EDITED", user_edit=True)
    preserved_char = TextElement(old_chars[1], page)
    manager._preserved_elements = {
        "words": [preserved_word],
        "chars": [preserved_char],
    }
    filtered_chars = manager._filter_preserved_native_chars(reloaded_chars)
    assert [manager._char_origin_key(char) for char in filtered_chars] == [
        manager._char_origin_key(reloaded_chars[0])
    ]
    generated_words = [make_word(char) for char in filtered_chars]
    elements_data = {
        "chars": [],
        "words": generated_words,
        "rects": [],
        "lines": [],
        "images": [],
        "regions": [],
    }
    remaining_raw = manager._merge_preserved_elements(elements_data, filtered_chars)

    assert remaining_raw == []
    assert preserved_word in elements_data["words"]
    assert generated_words[0] in elements_data["words"]
    assert len(elements_data["words"]) == 2


def test_source_based_removal_uses_text_element_pdf_default(mutable_pdf):
    page = mutable_pdf.pages[0]
    element = TextElement(
        {
            "text": "manual-native",
            "x0": 10,
            "top": 10,
            "x1": 80,
            "bottom": 20,
            "object_type": "word",
        },
        page,
    )
    assert element.source == "pdf"
    assert page.add_element(element, "words")

    assert page.remove_elements_by_source("words", "pdf") >= 1
    assert element not in page.words


def test_index_only_word_stays_coherent_across_char_edit_and_removal(mutable_pdf):
    page = mutable_pdf.pages[0]
    word = next(
        candidate
        for candidate in page.words
        if len(candidate.text) >= 4 and len(candidate._char_indices) == len(candidate.text)
    )
    original = word.text
    word._char_dicts = []
    first = word.chars[0]
    replacement = "Q" if first.text != "Q" else "Z"

    first.text = replacement

    assert word.text == replacement + original[1:]
    assert "".join(char.text for char in word.chars) == word.text

    before_removal = word.text
    word._char_dicts = []
    removed = word.chars[1]
    assert page.remove_element(removed)
    assert word.text == before_removal[0] + before_removal[2:]
    assert "".join(char.text for char in word.chars) == word.text
