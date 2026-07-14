import math
from types import SimpleNamespace

import pytest

from natural_pdf.exceptions import OCRError
from natural_pdf.ocr.ocr_cache import compute_cache_key
from natural_pdf.ocr.unified_dispatch import (
    EngineEntry,
    OCRRunResult,
    get_engine_cache,
    get_registry,
    register_engine,
)
from natural_pdf.services.ocr_service import OCRService


def _first_nonwhite_pixel(image):
    rgb = image.convert("RGB")
    for y in range(rgb.height):
        for x in range(rgb.width):
            if rgb.getpixel((x, y)) != (255, 255, 255):
                return x, y
    raise AssertionError("expected rendered PDF to contain a non-white pixel")


def test_page_ocr_exclusion_masks_rendered_pixels(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    clean = page.render(resolution=72)
    x, y = _first_nonwhite_pixel(clean)
    page.add_exclusion(page.create_region(x, y, min(x + 4, page.width), min(y + 4, page.height)))

    render_kwargs = page.services.ocr._render_kwargs(page, apply_exclusions=True)
    masked = page.render(resolution=72, **render_kwargs).convert("RGB")

    assert masked.getpixel((x, y)) == (255, 255, 255)
    assert render_kwargs["_ocr_exclusion_bboxes"]


def test_region_ocr_exclusion_uses_crop_relative_pixels(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    page_x, page_y = _first_nonwhite_pixel(page.render(resolution=72))
    left = max(0, page_x - 8)
    top = max(0, page_y - 8)
    region = page.create_region(
        left, top, min(page.width, page_x + 8), min(page.height, page_y + 8)
    )
    clean = region.render(resolution=72, crop=True, crop_bbox=region.bbox)
    pixel_x = page_x - int(left)
    pixel_y = page_y - int(top)
    assert clean.convert("RGB").getpixel((pixel_x, pixel_y)) != (255, 255, 255)
    exclusion = page.create_region(page_x, page_y, page_x + 4, page_y + 4)
    page.add_exclusion(exclusion)

    render_kwargs = region.services.ocr._render_kwargs(region, apply_exclusions=True)
    masked = region.render(resolution=72, **render_kwargs).convert("RGB")

    assert masked.getpixel((pixel_x, pixel_y)) == (255, 255, 255)
    assert render_kwargs["apply_exclusions"] is True


def test_unified_engine_receives_masked_pixels(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    pixel_x, pixel_y = _first_nonwhite_pixel(page.render(resolution=72))
    page.add_exclusion(page.create_region(pixel_x, pixel_y, pixel_x + 4, pixel_y + 4))
    crop_left = max(0, pixel_x - 8)
    crop_top = max(0, pixel_y - 8)
    region = page.create_region(
        crop_left,
        crop_top,
        page.width + 20,
        page.height + 20,
    )
    engine_image = _capture_engine_image(region)
    local_pixel = (pixel_x - int(crop_left), pixel_y - int(crop_top))
    assert engine_image.convert("RGB").getpixel(local_pixel) == (255, 255, 255)


@pytest.mark.parametrize("exclusion_form", ["direct", "selector"])
def test_element_method_exclusions_mask_actual_engine_image(exclusion_form, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    word = page.words[0]
    if exclusion_form == "direct":
        page.add_exclusion(word, method="element")
    else:
        page.add_exclusion("text", method="element")

    engine_image = _capture_engine_image(page).convert("RGB")
    x0, top, x1, bottom = word.bbox
    pixel_bbox = (
        math.floor(x0),
        math.floor(top),
        math.ceil(x1),
        math.ceil(bottom),
    )

    assert engine_image.crop(pixel_bbox).getextrema() == (
        (255, 255),
        (255, 255),
        (255, 255),
    )


@pytest.mark.parametrize(
    ("bbox", "effective_bbox", "expected_offset"),
    [
        ((-20, 10, 40, 50), (0.0, 10.0, 40.0, 50.0), (0.0, 10.0)),
        ((60, 10, 120, 50), (60.0, 10.0, 100.0, 50.0), (60.0, 10.0)),
        ((10, -20, 50, 40), (10.0, 0.0, 50.0, 40.0), (10.0, 0.0)),
        ((10, 60, 50, 120), (10.0, 60.0, 50.0, 100.0), (10.0, 60.0)),
    ],
)
def test_out_of_bounds_region_uses_one_effective_crop(
    monkeypatch, bbox, effective_bbox, expected_offset
):
    manager = _RecordingManager()
    page = SimpleNamespace(width=100, height=100, index=0)
    region = _RegionHost(page, bbox, manager)
    service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
    captured = {}

    def fake_run_ocr(**kwargs):
        captured.update(kwargs)
        x0, y0, x1, y1 = kwargs["render_kwargs"]["crop_bbox"]
        return OCRRunResult(
            results=[{"bbox": [10, 10, 30, 30], "text": "partial", "confidence": 0.9}],
            image_size=(int(2 * (x1 - x0)), int(2 * (y1 - y0))),
        )

    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)
    service.apply_ocr(
        region,
        engine="rapidocr",
        languages=["en"],
        device="cpu",
        replace="none",
        use_cache=False,
    )

    assert captured["render_kwargs"]["crop_bbox"] == effective_bbox
    call = manager.created[0]
    assert (call["offset_x"], call["offset_y"]) == expected_offset
    assert call["scale_x"] == 0.5
    assert call["scale_y"] == 0.5


@pytest.mark.parametrize(
    "bbox",
    [
        (-20, 10, -10, 40),
        (110, 10, 120, 40),
        (10, -20, 40, -10),
        (10, 110, 40, 120),
        (10, 20, 10, 40),
    ],
)
def test_zero_area_ocr_crop_fails_before_dispatch(monkeypatch, bbox):
    page = SimpleNamespace(width=100, height=100, index=0)
    region = _RegionHost(page, bbox, _RecordingManager())
    service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
    dispatched = {"value": False}

    def fail_if_dispatched(**kwargs):
        dispatched["value"] = True
        raise AssertionError("zero-area crop must fail before OCR dispatch")

    monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fail_if_dispatched)

    with pytest.raises(ValueError, match="OCR crop has no area within the page bounds"):
        service.apply_ocr(
            region,
            engine="rapidocr",
            languages=["en"],
            device="cpu",
            replace="none",
            use_cache=False,
        )
    assert dispatched["value"] is False


def test_extract_ocr_elements_is_pure_for_classic_payload(monkeypatch, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    words_before = list(page.words)
    chars_before = list(page.chars)
    regions_before = list(page.iter_regions())
    revision_before = page._text_state_version

    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[{"bbox": [10, 20, 50, 30], "text": "detached", "confidence": 0.9}],
            image_size=(int(page.width), int(page.height)),
            engine_type="classic",
        ),
    )

    extracted = page.extract_ocr_elements(engine="rapidocr", languages=["en"], device="cpu")

    assert [element.text for element in extracted] == ["detached"]
    assert list(page.words) == words_before
    assert list(page.chars) == chars_before
    assert list(page.iter_regions()) == regions_before
    assert page._text_state_version == revision_before


def test_extract_rejects_mixed_classic_payload_before_conversion(monkeypatch, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    words_before = list(page.words)
    chars_before = list(page.chars)
    revision_before = page._text_state_version
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[
                {"bbox": [10, 20, 50, 30], "text": "valid", "confidence": 0.9},
                {"text": "missing bbox", "confidence": 0.8},
            ],
            image_size=(int(page.width), int(page.height)),
            engine_type="classic",
        ),
    )

    with pytest.raises(OCRError, match=r"result 1.*bbox"):
        page.extract_ocr_elements(engine="rapidocr", languages=["en"], device="cpu")

    assert list(page.words) == words_before
    assert list(page.chars) == chars_before
    assert page._text_state_version == revision_before


def test_extract_materializes_valid_one_shot_bbox_before_reuse(monkeypatch, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[
                {
                    "bbox": (value for value in [10, 20, 50, 30]),
                    "text": "materialized",
                    "confidence": "0.9",
                }
            ],
            image_size=(int(page.width), int(page.height)),
            engine_type="classic",
        ),
    )

    elements = page.extract_ocr_elements(engine="rapidocr", languages=["en"], device="cpu")

    assert [element.text for element in elements] == ["materialized"]
    assert elements[0].confidence == pytest.approx(0.9)


def test_extract_vlm_table_payload_does_not_register_regions(monkeypatch, practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    regions_before = list(page.iter_regions())
    selector_regions_before = list(page.find_all("region", apply_exclusions=False))
    revision_before = page._text_state_version

    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **kwargs: OCRRunResult(
            results=[
                {
                    "bbox": [0, 0, 500, 300],
                    "text": "a\tb",
                    "confidence": 1.0,
                    "source_category": "table",
                    "raw_html": "<table><tr><td>a</td><td>b</td></tr></table>",
                },
                {
                    "bbox": [0, 350, 500, 400],
                    "text": "caption",
                    "confidence": 1.0,
                    "source_category": "text",
                },
            ],
            image_size=(1000, 1000),
            engine_type="vlm",
        ),
    )

    extracted = page.extract_ocr_elements(engine="vlm", languages=["en"], device="cpu")

    assert [element.text for element in extracted] == ["caption"]
    assert list(page.iter_regions()) == regions_before
    assert list(page.find_all("region", apply_exclusions=False)) == selector_regions_before
    assert page._text_state_version == revision_before


@pytest.mark.parametrize(("confidence", "expected"), [(None, None), ("0.9", 0.9)])
def test_extract_vlm_accepts_supported_confidence_forms(
    monkeypatch,
    practice_pdf_fresh,
    confidence,
    expected,
):
    page = practice_pdf_fresh.pages[0]
    monkeypatch.setattr(
        "natural_pdf.services.ocr_service.run_ocr",
        lambda **_: OCRRunResult(
            results=[
                {
                    "bbox": [0, 350, 500, 400],
                    "text": "caption",
                    "confidence": confidence,
                    "source_category": "text",
                }
            ],
            image_size=(1000, 1000),
            engine_type="vlm",
        ),
    )

    extracted = page.extract_ocr_elements(
        engine="vlm",
        languages=["en"],
        device="cpu",
        min_confidence=0.5,
    )

    assert [element.text for element in extracted] == ["caption"]
    assert extracted[0].confidence == expected


def test_exclusion_geometry_fingerprint_is_stable_and_cache_sensitive(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    state = {"x": 10}
    calls = {"count": 0}

    def dynamic_exclusion(current_page):
        calls["count"] += 1
        return current_page.create_region(state["x"], 20, state["x"] + 30, 40)

    page.add_exclusion(dynamic_exclusion)
    service = page.services.ocr
    first = service._render_kwargs(page, apply_exclusions=True)
    second = service._render_kwargs(page, apply_exclusions=True)
    assert service._exclusion_fingerprint(first) == service._exclusion_fingerprint(second)

    state["x"] = 15
    changed = service._render_kwargs(page, apply_exclusions=True)
    assert service._exclusion_fingerprint(first) != service._exclusion_fingerprint(changed)
    assert calls["count"] == 3

    base = dict(
        pdf_path="test.pdf",
        file_mtime_ns=1,
        file_size=1,
        page_index=0,
        engine_name="rapidocr",
        languages=("en",),
        resolution=150,
        detect_only=False,
        device="cpu",
    )
    first_key = compute_cache_key(
        **base, exclusion_geometry_key=service._exclusion_fingerprint(first)
    )
    same_key = compute_cache_key(
        **base, exclusion_geometry_key=service._exclusion_fingerprint(second)
    )
    changed_key = compute_cache_key(
        **base, exclusion_geometry_key=service._exclusion_fingerprint(changed)
    )
    assert first_key == same_key
    assert first_key != changed_key


class _RecordingManager:
    def __init__(self):
        self.created = []

    def create_text_elements_from_ocr(self, results, **kwargs):
        self.created.append({"results": results, **kwargs})
        return [SimpleNamespace(text=result.get("text")) for result in results]

    def clear_text_layer(self):
        return 0, 0

    def remove_ocr_elements(self):
        return 0


class _CapturingEngine:
    def __init__(self):
        self.images = []

    def is_available(self):
        return True

    def process_image(self, image, **kwargs):
        self.images.append(image.copy())
        return []


def _capture_engine_image(target):
    engine = _CapturingEngine()
    engine_name = "capture-ocr-mask"
    registry = get_registry()
    previous_entry = registry.get(engine_name)
    register_engine(
        engine_name,
        EngineEntry(engine_type="classic", provider=engine, needs_gpu_lock=False),
    )
    get_engine_cache().clear()

    try:
        target.extract_ocr_elements(
            engine=engine_name,
            resolution=72,
            languages=["en"],
            device="cpu",
        )
    finally:
        get_engine_cache().clear()
        if previous_entry is None:
            registry.pop(engine_name, None)
        else:
            registry[engine_name] = previous_entry

    assert engine.images
    return engine.images[0]


class _RegionHost:
    def __init__(self, page, bbox, manager):
        self.page = page
        self.bbox = bbox
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self._manager = manager

    def _ocr_scope(self):
        return "region"

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {
            "crop": True,
            "crop_bbox": self.bbox,
            "apply_exclusions": apply_exclusions,
        }

    def _ocr_element_manager(self):
        return self._manager
