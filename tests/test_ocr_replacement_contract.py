import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.region import FlowRegion
from natural_pdf.ocr.replacement import normalize_ocr_replace_mode
from natural_pdf.ocr.unified_dispatch import OCRRunResult
from natural_pdf.ocr.vlm_ocr import create_table_regions_from_ocr


@pytest.mark.parametrize("mode", ["ocr", "all", "none"])
def test_normalize_ocr_replace_mode_accepts_documented_modes(mode):
    assert normalize_ocr_replace_mode(mode) == mode


def test_normalize_ocr_replace_mode_normalizes_case_and_whitespace():
    assert normalize_ocr_replace_mode("  OCR  ") == "ocr"


@pytest.mark.parametrize("legacy", [True, False, None, 1])
def test_normalize_ocr_replace_mode_rejects_ambiguous_non_strings(legacy):
    with pytest.raises(TypeError, match="boolean replacement values are no longer supported"):
        normalize_ocr_replace_mode(legacy)


def test_normalize_ocr_replace_mode_rejects_unknown_string():
    with pytest.raises(ValueError, match="'ocr', 'all', or 'none'"):
        normalize_ocr_replace_mode("native")


@pytest.mark.parametrize("legacy", [True, False, None, 1])
def test_public_single_region_and_flow_entries_reject_legacy_replace_immediately(
    practice_pdf_fresh, legacy
):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    flow_region = FlowRegion.__new__(FlowRegion)
    flow_region.constituent_regions = [region]

    with pytest.raises(TypeError, match="replace must be one of"):
        page.apply_ocr(replace=legacy)
    with pytest.raises(TypeError, match="replace must be one of"):
        region.apply_ocr(replace=legacy)
    with pytest.raises(TypeError, match="replace must be one of"):
        flow_region.apply_ocr(replace=legacy)


def test_page_and_region_reject_non_callable_custom_ocr(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)

    with pytest.raises(TypeError, match="function must be callable"):
        page.apply_ocr(function="not-callable")
    with pytest.raises(TypeError):
        region.apply_ocr(ocr_function="not-callable")


def test_custom_ocr_rejects_detect_only_before_invoking_callback(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    calls = []

    def custom_ocr(target):
        calls.append(target)
        return "recognized"

    with pytest.raises(ValueError, match="function OCR cannot be combined.*detect_only"):
        page.apply_ocr(function=custom_ocr, detect_only=True)
    with pytest.raises(ValueError, match="function OCR cannot be combined.*detect_only"):
        region.apply_ocr(function=custom_ocr, detect_only=True)

    assert calls == []


def test_collection_rejects_detect_only_custom_before_work(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "ocr", "confidence": 1.0}],
        engine_name="test",
    )
    collection = ElementCollection([_ocr_words(page)[0]])
    calls = []

    with pytest.raises(ValueError, match="function OCR cannot be combined.*detect_only"):
        collection.apply_ocr(
            function=lambda target: calls.append(target) or "recognized",
            detect_only=True,
        )

    assert calls == []


def _disable_ocr_cache(monkeypatch):
    monkeypatch.setattr("natural_pdf.ocr.ocr_cache.get_default_cache", lambda: None)


def _ocr_words(page):
    return [word for word in page.words if getattr(word, "source", None) == "ocr"]


def test_region_ocr_replacement_is_geometry_scoped(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    midpoint = page.width / 2
    page.create_text_elements_from_ocr(
        [
            {"bbox": [10, 10, 40, 30], "text": "left-old", "confidence": 1.0},
            {
                "bbox": [midpoint + 10, 10, midpoint + 50, 30],
                "text": "right-old",
                "confidence": 1.0,
            },
        ],
        engine_name="test",
    )
    left = page.create_region(0, 0, midpoint, page.height)

    removed = left.remove_ocr_elements()

    assert removed == 2  # one word and its linked character entry
    assert [word.text for word in _ocr_words(page)] == ["right-old"]


def test_region_clear_text_layer_preserves_out_of_scope_native_text(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    words_before = list(page.words)
    target = words_before[0]
    target_region = page.create_region(*target.bbox)
    outside_ids = {
        id(word) for word in words_before if not target_region.is_element_center_inside(word)
    }

    removed_words, removed_chars = target_region.clear_text_layer()

    assert removed_words >= 1
    assert removed_chars >= 1
    assert outside_ids <= {id(word) for word in page.words}


def test_scoped_removal_preserves_char_shared_with_retained_word(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    manager = page._element_mgr
    words = list(page.words)
    chars = list(page.chars)
    removed_word = words[0]
    retained_word = words[-1]
    shared_char_dict = removed_word._char_dicts[0]
    retained_word._char_dicts = [*retained_word._char_dicts, shared_char_dict]
    manager._reindex_words(words, chars)

    manager.remove_text_elements_in_bbox(removed_word.bbox)

    assert retained_word in page.words
    assert shared_char_dict in retained_word._char_dicts
    assert any(char._obj is shared_char_dict for char in page.chars)


def test_same_page_flow_replacement_does_not_erase_prior_constituent_output(
    monkeypatch, practice_pdf_fresh
):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    midpoint = page.width / 2
    left = page.create_region(0, 0, midpoint, page.height)
    right = page.create_region(midpoint, 0, page.width, page.height)
    flow_region = FlowRegion.__new__(FlowRegion)
    flow_region.constituent_regions = [left, right]

    def fake_run_ocr(**kwargs):
        target = kwargs["target"]
        label = "left-new" if target is left else "right-new"
        return OCRRunResult(
            results=[{"bbox": [10, 10, 90, 30], "text": label, "confidence": 1.0}],
            image_size=(100, 100),
            engine_type="classic",
        )

    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)

    flow_region.apply_ocr(engine="rapidocr", replace="ocr")

    assert {word.text for word in _ocr_words(page)} == {"left-new", "right-new"}


def test_overlapping_flow_replacement_preserves_outputs_from_the_same_call(
    monkeypatch, practice_pdf_fresh
):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    first = page.create_region(0, 0, page.width / 2, page.height / 2)
    second = page.create_region(0, 0, page.width / 2, page.height / 2)
    flow_region = FlowRegion.__new__(FlowRegion)
    flow_region.constituent_regions = [first, second]

    def fake_run_ocr(**kwargs):
        label = "first-new" if kwargs["target"] is first else "second-new"
        return OCRRunResult(
            results=[{"bbox": [10, 10, 90, 30], "text": label, "confidence": 1.0}],
            image_size=(100, 100),
            engine_type="classic",
        )

    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)

    flow_region.apply_ocr(engine="rapidocr", replace="ocr")

    assert {word.text for word in _ocr_words(page)} == {"first-new", "second-new"}


def test_overlapping_flow_detection_preserves_outputs_from_the_same_call(
    monkeypatch, practice_pdf_fresh
):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    first = page.create_region(0, 0, page.width / 2, page.height / 2)
    second = page.create_region(0, 0, page.width / 2, page.height / 2)
    flow_region = FlowRegion.__new__(FlowRegion)
    flow_region.constituent_regions = [first, second]
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[{"bbox": [10, 10, 90, 30], "text": "", "confidence": None}],
            image_size=(100, 100),
            engine_type="classic",
        ),
    )

    flow_region.apply_ocr(engine="rapidocr", detect_only=True)

    detections = [word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False)]
    assert len(detections) == 2


def test_detect_only_refreshes_scoped_artifacts_and_preserves_text(monkeypatch, practice_pdf_fresh):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "recognized", "confidence": 1.0}],
        engine_name="test",
    )
    outside_detection = page.create_text_elements_from_ocr(
        [
            {
                "bbox": [page.width * 0.75, 20, page.width * 0.9, 40],
                "text": "",
                "confidence": None,
                "_ocr_detection_only": True,
                "source_category": "detection",
            }
        ],
        engine_name="test",
    )[0]
    native_ids = {id(word) for word in page.words if getattr(word, "source", None) == "native"}
    char_count_before = len(page.chars)

    responses = iter(
        [
            [{"bbox": [10, 10, 40, 30], "text": "", "confidence": None}],
            [{"bbox": [50, 50, 90, 70], "text": "", "confidence": None}],
        ]
    )
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=next(responses),
            image_size=(100, 100),
            engine_type="classic",
        ),
    )

    region.apply_ocr(engine="rapidocr", detect_only=True)
    first_detection = next(
        word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False)
    )
    region.apply_ocr(engine="rapidocr", detect_only=True)

    assert "recognized" in {word.text for word in _ocr_words(page)}
    assert outside_detection in page.words
    assert native_ids <= {id(word) for word in page.words}
    detections = [word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False)]
    assert len(detections) == 1
    assert detections[0] is not first_detection
    assert detections[0].text == ""
    assert detections[0]._char_dicts == []
    assert detections[0]._obj["source_category"] == "detection"
    assert detections[0].metadata["ocr_detection_only"] is True
    assert len(page.chars) == char_count_before


@pytest.mark.parametrize("replace", ["all", "none"])
def test_detect_only_rejects_recognition_replacement_modes(practice_pdf_fresh, replace):
    page = practice_pdf_fresh.pages[0]

    with pytest.raises(ValueError, match="refreshes detection artifacts"):
        page.apply_ocr(detect_only=True, replace=replace)


def test_vlm_detect_only_creates_geometry_not_tables_or_chars(monkeypatch, practice_pdf_fresh):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    regions_before = list(page.iter_regions())
    char_count_before = len(page.chars)
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[
                {
                    "bbox": [10, 10, 90, 90],
                    "text": "recognized table text",
                    "confidence": 1.0,
                    "source_category": "table",
                }
            ],
            image_size=(100, 100),
            engine_type="vlm",
        ),
    )

    region.apply_ocr(engine="vlm", detect_only=True)

    detections = [word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False)]
    assert len(detections) == 1
    assert detections[0].text == ""
    assert len(page.chars) == char_count_before
    assert list(page.iter_regions()) == regions_before


def test_malformed_detection_payload_preserves_prior_detections(monkeypatch, practice_pdf_fresh):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    responses = iter(
        [
            [{"bbox": [10, 10, 90, 30], "text": "", "confidence": None}],
            [{"text": "missing geometry", "confidence": 1.0}],
        ]
    )
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=next(responses), image_size=(100, 100), engine_type="classic"
        ),
    )

    region.apply_ocr(engine="rapidocr", detect_only=True)
    original = next(word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False))
    region.apply_ocr(engine="rapidocr", detect_only=True)

    assert original in page.words
    assert [word for word in _ocr_words(page) if getattr(word, "is_ocr_detection", False)] == [
        original
    ]


def test_invalid_payload_does_not_remove_existing_ocr(monkeypatch, practice_pdf_fresh):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "keep-me", "confidence": 1.0}],
        engine_name="test",
    )
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[{"bbox": [1, 1, 2, 2], "text": "invalid", "confidence": 1.0}],
            image_size=(0, 0),
            engine_type="classic",
        ),
    )

    region.apply_ocr(engine="rapidocr", replace="ocr")

    assert "keep-me" in {word.text for word in _ocr_words(page)}


def test_detached_custom_ocr_never_removes_page_text(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "keep-me", "confidence": 1.0}],
        engine_name="test",
    )
    words_before = list(page.words)
    chars_before = list(page.chars)

    region.apply_custom_ocr(
        lambda _region: "detached",
        replace="all",
        add_to_page=False,
    )

    assert list(page.words) == words_before
    assert list(page.chars) == chars_before


def test_empty_custom_ocr_result_is_non_destructive(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "keep-me", "confidence": 1.0}],
        engine_name="test",
    )
    words_before = list(page.words)
    chars_before = list(page.chars)

    region.apply_custom_ocr(lambda _region: "  \n\t ", replace="all")

    assert list(page.words) == words_before
    assert list(page.chars) == chars_before


def test_page_function_ocr_is_attached_and_forwards_output_controls(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    with pytest.raises(TypeError):
        page.apply_ocr(function=lambda _region: "detached", add_to_page=False)

    page.apply_ocr(
        function=lambda _region: "attached",
        source_label="page-custom",
        confidence=0.42,
        replace="none",
    )
    created = [word for word in page.words if getattr(word, "ocr_engine", None) == "page-custom"]
    assert [word.text for word in created] == ["attached"]
    assert created[0].source == "ocr"
    assert created[0].confidence == pytest.approx(0.42)


def test_page_apply_custom_ocr_direct_method_supports_detached_and_attached_output(
    practice_pdf_fresh,
):
    page = practice_pdf_fresh.pages[0]
    words_before = list(page.words)
    chars_before = list(page.chars)

    page.apply_custom_ocr(
        ocr_function=lambda _region: "detached-direct",
        source_label="direct-custom",
        confidence=0.2,
        add_to_page=False,
        replace="all",
    )
    assert list(page.words) == words_before
    assert list(page.chars) == chars_before

    page.apply_custom_ocr(
        ocr_function=lambda _region: "attached-direct",
        source_label="direct-custom",
        confidence=0.2,
        add_to_page=True,
        replace="none",
    )
    created = [word for word in page.words if getattr(word, "ocr_engine", None) == "direct-custom"]
    assert [word.text for word in created] == ["attached-direct"]
    assert created[0].source == "ocr"
    assert created[0].confidence == pytest.approx(0.2)


def test_function_ocr_replaces_prior_function_output_across_labels(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height / 2)

    region.apply_ocr(function=lambda _region: "first", source_label="engine-a")
    region.apply_ocr(function=lambda _region: "second", source_label="engine-b")

    created = [word for word in _ocr_words(page) if word.text in {"first", "second"}]
    assert [word.text for word in created] == ["second"]
    assert created[0].ocr_engine == "engine-b"


def test_engine_ocr_replaces_prior_function_output(monkeypatch, practice_pdf_fresh):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height / 2)
    region.apply_ocr(function=lambda _region: "custom", source_label="engine-a")
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[{"bbox": [10, 10, 90, 30], "text": "builtin", "confidence": 1.0}],
            image_size=(100, 100),
            engine_type="classic",
        ),
    )

    region.apply_ocr(engine="rapidocr")

    assert "custom" not in {word.text for word in _ocr_words(page)}
    assert "builtin" in {word.text for word in _ocr_words(page)}


def test_malformed_later_vlm_table_never_removes_or_partially_registers(
    monkeypatch, practice_pdf_fresh
):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    page.create_text_elements_from_ocr(
        [{"bbox": [10, 10, 40, 30], "text": "keep-me", "confidence": 1.0}],
        engine_name="test",
    )
    regions_before = list(page.iter_regions())
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[
                {
                    "bbox": [10, 10, 90, 50],
                    "text": "valid\ttable",
                    "confidence": 1.0,
                    "source_category": "table",
                },
                {
                    "text": "missing bbox",
                    "confidence": 1.0,
                    "source_category": "table",
                },
            ],
            image_size=(100, 100),
            engine_type="vlm",
        ),
    )

    region.apply_ocr(engine="vlm", replace="ocr")

    assert "keep-me" in {word.text for word in _ocr_words(page)}
    assert list(page.iter_regions()) == regions_before


def test_table_region_conversion_validates_all_results_before_registration(
    practice_pdf_fresh,
):
    page = practice_pdf_fresh.pages[0]
    regions_before = list(page.iter_regions())
    results = [
        {
            "bbox": [10, 10, 90, 50],
            "text": "valid\ttable",
            "source_category": "table",
        },
        {"text": "missing bbox", "source_category": "table"},
    ]

    with pytest.raises(ValueError, match="valid bboxes"):
        create_table_regions_from_ocr(page, results)

    assert list(page.iter_regions()) == regions_before


def test_ocr_replace_removes_generated_vlm_tables_but_preserves_user_regions(
    monkeypatch, practice_pdf_fresh
):
    _disable_ocr_cache(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height)
    user_table = page.create_region(20, 20, 80, 80)
    user_table.region_type = "table"
    user_table.source = "vlm"
    page.add_region(user_table)

    payloads = iter(
        [
            OCRRunResult(
                results=[
                    {
                        "bbox": [10, 10, 90, 90],
                        "text": "a\tb",
                        "confidence": 1.0,
                        "source_category": "table",
                    }
                ],
                image_size=(100, 100),
                engine_type="vlm",
            ),
            OCRRunResult(
                results=[{"bbox": [10, 10, 90, 30], "text": "new", "confidence": 1.0}],
                image_size=(100, 100),
                engine_type="classic",
            ),
        ]
    )
    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", lambda **kwargs: next(payloads))

    region.apply_ocr(engine="vlm", replace="none")
    generated = [
        item for item in page.iter_regions() if getattr(item, "_natural_pdf_ocr_generated", False)
    ]
    assert len(generated) == 1

    region.apply_ocr(engine="rapidocr", replace="ocr")

    assert user_table in page.iter_regions()
    assert generated[0] not in page.iter_regions()
