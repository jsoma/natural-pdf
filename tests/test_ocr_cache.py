"""Tests for OCR result caching."""

import pytest

from natural_pdf.ocr.ocr_cache import OCRCache, compute_cache_key
from natural_pdf.ocr.unified_dispatch import OCRRunResult

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
