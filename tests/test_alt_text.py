"""Tests for Region.alt_text support in extract_text()."""

import pytest

from natural_pdf import PDF, options, set_option
from natural_pdf.elements.region import Region


@pytest.fixture
def pdf():
    p = PDF("pdfs/01-practice.pdf")
    yield p
    p.close()


@pytest.fixture
def page(pdf):
    return pdf.pages[0]


# ---- 1. Self short-circuit ----


def test_region_alt_text_self_shortcircuit(page):
    """Region with alt_text returns it directly from extract_text()."""
    region = page.create_region(50, 50, 100, 100)
    region.alt_text = "[CHECKED]"
    assert region.extract_text() == "[CHECKED]"


def test_region_alt_text_self_shortcircuit_return_textmap(page):
    """Self short-circuit also works when return_textmap=True."""
    region = page.create_region(50, 50, 100, 100)
    region.alt_text = "[UNCHECKED]"
    text, textmap = region.extract_text(return_textmap=True)
    assert text == "[UNCHECKED]"
    assert textmap is None


# ---- 2. Parent region picks up child alt_text ----


def test_parent_region_includes_child_alt_text(page):
    """A parent region includes alt_text from an overlapping child region."""
    # Create a child region with alt_text inside a parent region
    child = page.create_region(60, 60, 80, 80)
    child.alt_text = "[CHECKED]"
    page.add_region(child, source="test")

    # Create a larger parent region that encloses the child
    parent = page.create_region(50, 50, 200, 200)
    text = parent.extract_text()
    assert "[CHECKED]" in text

    page.remove_regions(source="test")


# ---- 3. Non-overlapping alt_text regions excluded ----


def test_non_overlapping_alt_text_excluded(page):
    """Alt_text regions outside the parent bbox are not included."""
    # Create a region with alt_text far from the parent
    far_region = page.create_region(500, 500, 550, 550)
    far_region.alt_text = "[SHOULD_NOT_APPEAR]"
    page.add_region(far_region, source="test")

    # Create a small parent region that doesn't overlap
    parent = page.create_region(50, 50, 100, 100)
    text = parent.extract_text()
    assert "[SHOULD_NOT_APPEAR]" not in text

    page.remove_regions(source="test")


# ---- 4. Page.extract_text() includes alt_text ----


def test_page_extract_text_includes_alt_text(page):
    """Page.extract_text() includes alt_text from page regions."""
    region = page.create_region(50, 50, 100, 100)
    region.alt_text = "[CHECKED]"
    page.add_region(region, source="test")

    text = page.extract_text()
    assert "[CHECKED]" in text

    page.remove_regions(source="test")


# ---- 5. options.alt_text defaults and set_option ----


def test_alt_text_option_defaults():
    """options.alt_text has correct defaults."""
    assert options.alt_text.checkbox_checked == "[CHECKED]"
    assert options.alt_text.checkbox_unchecked == "[UNCHECKED]"


def test_set_option_alt_text():
    """set_option works for alt_text config."""
    original_checked = options.alt_text.checkbox_checked
    original_unchecked = options.alt_text.checkbox_unchecked
    try:
        set_option("alt_text.checkbox_checked", "YES")
        set_option("alt_text.checkbox_unchecked", "NO")
        assert options.alt_text.checkbox_checked == "YES"
        assert options.alt_text.checkbox_unchecked == "NO"
    finally:
        set_option("alt_text.checkbox_checked", original_checked)
        set_option("alt_text.checkbox_unchecked", original_unchecked)


# ---- 6. detect_checkboxes() auto-sets alt_text ----


def test_detect_checkboxes_sets_alt_text(page, monkeypatch):
    """detect_checkboxes() sets alt_text based on classification."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

    # Create fake regions simulating detection output
    fake_regions = []
    r1 = page.create_region(10, 10, 30, 30)
    r1.region_type = "checkbox"
    r1.normalized_type = "checkbox"
    r1.is_checked = True
    r1.checkbox_state = "checked"
    r1.confidence = 0.95
    r1.source = "checkbox"
    r1.analyses = {"checkbox": {}}
    fake_regions.append(r1)

    r2 = page.create_region(10, 40, 30, 60)
    r2.region_type = "checkbox"
    r2.normalized_type = "checkbox"
    r2.is_checked = False
    r2.checkbox_state = "unchecked"
    r2.confidence = 0.90
    r2.source = "checkbox"
    r2.analyses = {"checkbox": {}}
    fake_regions.append(r2)

    analyzer = CheckboxAnalyzer(page)

    # Monkeypatch to skip actual detection and classification
    monkeypatch.setattr(analyzer, "_build_options", lambda *a, **kw: None)
    monkeypatch.setattr(analyzer, "_detect_auto", lambda opts: [])
    monkeypatch.setattr(analyzer, "_detect_with_engine", lambda name, opts: [])

    # Directly call the flow that sets alt_text by simulating what
    # detect_checkboxes does after classification
    from natural_pdf.analyzers.checkbox.classifier import CheckboxClassifier

    # We'll inject our regions directly. The alt_text assignment happens
    # in detect_checkboxes after classification, so let's call the analyzer
    # with a monkeypatched pipeline that returns our fake_regions.
    def fake_detect(opts):
        return [
            {
                "bbox": (10, 10, 30, 30),
                "is_checked": True,
                "checkbox_state": "checked",
                "confidence": 0.95,
                "coord_space": "pdf",
            },
            {
                "bbox": (10, 40, 30, 60),
                "is_checked": False,
                "checkbox_state": "unchecked",
                "confidence": 0.90,
                "coord_space": "pdf",
            },
        ]

    monkeypatch.setattr(analyzer, "_detect_auto", fake_detect)
    monkeypatch.setattr(analyzer, "_filter_text_regions", lambda regions: regions)
    monkeypatch.setattr(
        CheckboxClassifier,
        "classify_regions",
        staticmethod(lambda regions, page, **kw: None),
    )

    results = analyzer.detect_checkboxes()
    checked = [r for r in results if r.is_checked is True]
    unchecked = [r for r in results if r.is_checked is False]

    assert len(checked) >= 1
    assert checked[0].alt_text == "[CHECKED]"
    assert len(unchecked) >= 1
    assert unchecked[0].alt_text == "[UNCHECKED]"


# ---- 7. Unknown checkbox state leaves alt_text as None ----


def test_unknown_checkbox_state_no_alt_text(page, monkeypatch):
    """When is_checked is None, alt_text stays None."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer
    from natural_pdf.analyzers.checkbox.classifier import CheckboxClassifier

    analyzer = CheckboxAnalyzer(page)

    def fake_detect(opts):
        return [
            {
                "bbox": (10, 10, 30, 30),
                "is_checked": None,
                "checkbox_state": "unknown",
                "confidence": 0.5,
                "coord_space": "pdf",
            },
        ]

    monkeypatch.setattr(analyzer, "_build_options", lambda *a, **kw: None)
    monkeypatch.setattr(analyzer, "_detect_auto", fake_detect)
    monkeypatch.setattr(analyzer, "_filter_text_regions", lambda regions: regions)
    # Classify_regions is called but does nothing here
    monkeypatch.setattr(
        CheckboxClassifier,
        "classify_regions",
        staticmethod(lambda regions, page, **kw: None),
    )

    results = analyzer.detect_checkboxes()
    assert len(results) == 1
    assert results[0].alt_text is None


# ---- 8. Re-detection idempotency ----


def test_redetection_replaces_old_regions(page, monkeypatch):
    """Re-running detect_checkboxes replaces old regions (no duplicates)."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer
    from natural_pdf.analyzers.checkbox.classifier import CheckboxClassifier

    analyzer = CheckboxAnalyzer(page)

    def fake_detect(opts):
        return [
            {
                "bbox": (10, 10, 30, 30),
                "is_checked": True,
                "checkbox_state": "checked",
                "confidence": 0.95,
                "coord_space": "pdf",
            },
        ]

    monkeypatch.setattr(analyzer, "_build_options", lambda *a, **kw: None)
    monkeypatch.setattr(analyzer, "_detect_auto", fake_detect)
    monkeypatch.setattr(analyzer, "_filter_text_regions", lambda regions: regions)
    monkeypatch.setattr(
        CheckboxClassifier,
        "classify_regions",
        staticmethod(lambda regions, page, **kw: None),
    )

    results1 = analyzer.detect_checkboxes()
    assert len(results1) == 1
    assert results1[0].alt_text == "[CHECKED]"

    # Run again — old regions should be replaced
    results2 = analyzer.detect_checkboxes()
    assert len(results2) == 1

    # Count checkbox regions on the page
    checkbox_regions = [
        r for r in page.iter_regions() if getattr(r, "region_type", None) == "checkbox"
    ]
    assert len(checkbox_regions) == 1
    assert checkbox_regions[0].alt_text == "[CHECKED]"


# ---- Region.alt_text default ----


def test_region_alt_text_default_none(page):
    """Newly created regions have alt_text=None."""
    region = page.create_region(10, 10, 100, 100)
    assert region.alt_text is None


def test_region_no_alt_text_extract_text_unchanged(page):
    """Without alt_text, extract_text() behaves normally."""
    region = page.create_region(10, 10, 100, 100)
    # Should not raise and should return whatever the normal path returns
    text = region.extract_text()
    assert isinstance(text, str)


# ---- 9. alt_text suppresses underlying text ----


def test_alt_text_suppresses_underlying_text(page):
    """When a child region has alt_text, it replaces (not duplicates) underlying chars."""
    # Find a word on the page to use as a target
    word = page.find("text")
    assert word is not None, "Test PDF must have at least one text element"
    original_text = word.text

    # Create a child region exactly covering the word, with alt_text
    child = page.create_region(word.x0, word.top, word.x1, word.bottom)
    child.alt_text = "[REPLACED]"
    page.add_region(child, source="test")

    # Create a parent region that encloses the word
    parent = page.create_region(word.x0 - 1, word.top - 1, word.x1 + 1, word.bottom + 1)
    text = parent.extract_text()

    # The replacement text must appear
    assert "[REPLACED]" in text
    # The original word text must NOT appear (suppression, not addition)
    assert original_text not in text

    page.remove_regions(source="test")


# ---- 10. Unicode checkbox detection ----


def test_unicode_checkbox_detection(page, monkeypatch):
    """_detect_unicode_checkboxes finds Unicode ballot box characters."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer
    from natural_pdf.elements.text import TextElement

    # Create fake word elements for Unicode checkboxes
    fake_words = []
    for i, (char, should_be_checked) in enumerate(
        [("\u2610", False), ("\u2611", True), ("\u2612", True)]
    ):
        obj = {
            "text": char,
            "x0": 10.0,
            "top": 10.0 + i * 20,
            "x1": 20.0,
            "bottom": 20.0 + i * 20,
            "width": 10.0,
            "height": 10.0,
            "object_type": "text",
            "page_number": 0,
            "fontname": "TestFont",
            "size": 12.0,
            "upright": True,
            "direction": 1,
            "source": "pdf",
            "_char_dicts": [],
        }
        elem = TextElement(obj, page)
        fake_words.append(elem)

    monkeypatch.setattr(type(page), "words", property(lambda self: fake_words))

    analyzer = CheckboxAnalyzer(page)
    regions = analyzer._detect_unicode_checkboxes()

    assert len(regions) == 3
    assert regions[0].is_checked is False
    assert regions[0].checkbox_state == "unchecked"
    assert regions[0].confidence == 1.0
    assert regions[0].model == "unicode"
    assert regions[1].is_checked is True
    assert regions[2].is_checked is True


# ---- 11. Unicode checkbox dedup ----


def test_unicode_checkbox_dedup(page):
    """When visual and Unicode regions overlap, visual region wins."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

    analyzer = CheckboxAnalyzer(page)

    # Create a Unicode region
    ur = page.create_region(10, 10, 20, 20)
    ur.is_checked = False

    # Create a visual region overlapping the same spot
    vr = page.create_region(8, 8, 22, 22)
    vr.is_checked = True

    result = analyzer._dedup_unicode([ur], [vr])
    assert len(result) == 0, "Unicode region should be deduped when visual region overlaps"


def test_unicode_checkbox_dedup_no_overlap(page):
    """Non-overlapping Unicode regions survive dedup."""
    from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

    analyzer = CheckboxAnalyzer(page)

    ur = page.create_region(100, 100, 110, 110)
    ur.is_checked = False

    vr = page.create_region(10, 10, 20, 20)
    vr.is_checked = True

    result = analyzer._dedup_unicode([ur], [vr])
    assert len(result) == 1, "Non-overlapping Unicode region should survive dedup"
