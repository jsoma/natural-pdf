"""Tests for OCR result caching."""

from types import SimpleNamespace

from natural_pdf.ocr.ocr_cache import OCRCache, compute_cache_key, set_default_cache
from natural_pdf.ocr.unified_dispatch import OCRRunResult
from natural_pdf.services.ocr_service import OCRService

# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestCacheKey:
    """compute_cache_key determinism and sensitivity."""

    BASE = dict(
        pdf_path="test.pdf",
        file_mtime_ns=1000000000,
        file_size=5000,
        page_index=0,
        engine_name="rapidocr",
        languages=("en",),
        resolution=150,
        detect_only=False,
        device="cpu",
        options_init_key="",
        apply_exclusions=True,
        model=None,
        prompt=None,
        instructions=None,
        max_new_tokens=None,
    )

    def test_deterministic(self):
        """Same inputs produce the same key."""
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**self.BASE)
        assert key1 == key2

    def test_changes_with_mtime(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "file_mtime_ns": 2000000000})
        assert key1 != key2

    def test_changes_with_engine(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "engine_name": "easyocr"})
        assert key1 != key2

    def test_changes_with_page_index(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "page_index": 1})
        assert key1 != key2

    def test_changes_with_resolution(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "resolution": 300})
        assert key1 != key2

    def test_changes_with_languages(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "languages": ("en", "fr")})
        assert key1 != key2

    def test_changes_with_crop_bbox(self):
        key1 = compute_cache_key(**self.BASE)
        key2 = compute_cache_key(**{**self.BASE, "crop_bbox": (10, 20, 110, 120)})
        assert key1 != key2


# ---------------------------------------------------------------------------
# Cache store / retrieve tests
# ---------------------------------------------------------------------------


class TestOCRCache:
    """OCRCache put/get/clear operations."""

    def _make_result(self, text="Hello"):
        return OCRRunResult(
            results=[{"bbox": (100, 200, 300, 250), "text": text, "confidence": 0.95}],
            image_size=(1000, 1000),
            engine_type="classic",
        )

    def test_put_and_get(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        result = self._make_result()
        cache.put("test_key", result, "rapidocr", 0)

        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert len(retrieved.results) == 1
        assert retrieved.results[0]["text"] == "Hello"
        assert retrieved.image_size == (1000, 1000)
        assert retrieved.engine_type == "classic"

    def test_miss_returns_none(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        assert cache.get("nonexistent") is None

    def test_clear(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        result = self._make_result()
        cache.put("key1", result, "rapidocr", 0)
        cache.put("key2", result, "rapidocr", 1)

        removed = cache.clear()
        assert removed == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_empty_cache(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        assert cache.clear() == 0

    def test_multiple_entries(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        r1 = self._make_result("First")
        r2 = self._make_result("Second")
        cache.put("k1", r1, "rapidocr", 0)
        cache.put("k2", r2, "rapidocr", 1)

        assert cache.get("k1").results[0]["text"] == "First"
        assert cache.get("k2").results[0]["text"] == "Second"

    def test_overwrite(self, tmp_path):
        cache = OCRCache(cache_dir=tmp_path)
        r1 = self._make_result("Old")
        r2 = self._make_result("New")
        cache.put("key", r1, "rapidocr", 0)
        cache.put("key", r2, "rapidocr", 0)

        assert cache.get("key").results[0]["text"] == "New"

    def test_bbox_tuple_round_trip(self, tmp_path):
        """Bounding boxes stored as lists should be retrievable."""
        cache = OCRCache(cache_dir=tmp_path)
        result = OCRRunResult(
            results=[{"bbox": (10, 20, 30, 40), "text": "test", "confidence": 0.9}],
            image_size=(500, 500),
        )
        cache.put("bbox_test", result, "rapidocr", 0)

        retrieved = cache.get("bbox_test")
        # JSON round-trip converts tuples to lists
        assert retrieved.results[0]["bbox"] == [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# Service integration regression tests
# ---------------------------------------------------------------------------


class _RecordingOCRManager:
    def __init__(self):
        self.created = []

    def remove_ocr_elements(self):
        return 0

    def clear_text_layer(self):
        return (0, 0)

    def create_text_elements_from_ocr(
        self,
        ocr_results,
        scale_x=None,
        scale_y=None,
        offset_x=0.0,
        offset_y=0.0,
        engine_name=None,
    ):
        call = {
            "ocr_results": ocr_results,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "engine_name": engine_name,
        }
        self.created.append(call)
        return [SimpleNamespace(text=result.get("text")) for result in ocr_results]


class _FakePage:
    width = 100
    height = 100
    index = 0

    def __init__(self, pdf_path):
        self.pdf = SimpleNamespace(_resolved_path=str(pdf_path))
        self.manager = _RecordingOCRManager()

    def _ocr_element_manager(self):
        return self.manager

    def _ocr_scope(self):
        return "page"

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {"apply_exclusions": apply_exclusions}


class _FakeRegion:
    width = 50
    height = 50
    bbox = (25, 25, 75, 75)

    def __init__(self, page):
        self.page = page
        self.manager = _RecordingOCRManager()

    def _ocr_element_manager(self):
        return self.manager

    def _ocr_scope(self):
        return "region"

    def _ocr_render_kwargs(self, *, apply_exclusions=True):
        return {"crop": True}


def test_region_ocr_cache_isolated_from_full_page_payload(monkeypatch, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    previous_cache = set_default_cache(OCRCache(cache_dir=tmp_path / "ocr-cache"))
    try:
        service = OCRService(SimpleNamespace(get_option=lambda *args, **kwargs: None))
        page = _FakePage(pdf_path)
        region = _FakeRegion(page)
        calls = []

        def fake_run_ocr(**kwargs):
            calls.append(kwargs)
            target = kwargs["target"]
            if target is page:
                return OCRRunResult(
                    results=[{"bbox": [80, 80, 100, 100], "text": "outside", "confidence": 0.99}],
                    image_size=(100, 100),
                )
            if target is region:
                return OCRRunResult(
                    results=[{"bbox": [0, 0, 10, 10], "text": "inside", "confidence": 0.99}],
                    image_size=(50, 50),
                )
            raise AssertionError(f"Unexpected OCR target: {target!r}")

        monkeypatch.setattr("natural_pdf.services.ocr_service.run_ocr", fake_run_ocr)

        service.apply_ocr(page, engine="rapidocr", languages=["en"], device="cpu")
        service.apply_ocr(region, engine="rapidocr", languages=["en"], device="cpu")

        assert [call["target"] for call in calls] == [page, region]
        assert calls[1]["render_kwargs"] == {"crop": True}

        region_create_call = region.manager.created[-1]
        assert region_create_call["ocr_results"][0]["text"] == "inside"
        assert region_create_call["scale_x"] == 1.0
        assert region_create_call["scale_y"] == 1.0
        assert region_create_call["offset_x"] == 25
        assert region_create_call["offset_y"] == 25
    finally:
        set_default_cache(previous_cache)
