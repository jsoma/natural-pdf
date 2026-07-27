"""Tests for the Round 1 hardening fixes: silent-failure surfacing and
thread-safety of the registry/context/engine-cache layers."""

import io
import logging
import threading

import pytest

from natural_pdf import PDF
from natural_pdf.core.context import PDFContext
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.ocr.unified_dispatch import EngineCache
from natural_pdf.services.qa_service import QAService

PRACTICE_PDF = "pdfs/01-practice.pdf"


# ---------------------------------------------------------------------------
# Silent-failure surfacing
# ---------------------------------------------------------------------------


class _TellFailsOnceStream(io.BytesIO):
    """Stream whose first tell() raises, breaking byte capture but not loading."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self._tell_calls = 0

    def tell(self) -> int:
        self._tell_calls += 1
        if self._tell_calls == 1:
            raise OSError("tell not supported")
        return super().tell()


def test_stream_byte_capture_failure_warns(caplog):
    with open(PRACTICE_PDF, "rb") as fh:
        data = fh.read()

    stream = _TellFailsOnceStream(data)
    with caplog.at_level(logging.WARNING, logger="natural_pdf.core.pdf"):
        pdf = PDF(stream)
    try:
        assert any(
            "Could not capture stream bytes" in rec.message for rec in caplog.records
        ), "expected a warning when stream byte capture fails"
        # The PDF must still be usable.
        assert len(pdf.pages) > 0
        assert pdf.pages[0].extract_text()
    finally:
        pdf.close()


def test_exclusion_label_collision_warns(caplog):
    pdf = PDF(PRACTICE_PDF)
    try:
        page = pdf.pages[0]
        pdf.add_exclusion(lambda p: p.create_region(0, 0, p.width, 20), label="header")
        page.add_exclusion(page.create_region(0, 0, page.width, 10), label="header")

        with caplog.at_level(logging.WARNING, logger="natural_pdf.core.page"):
            page._get_exclusion_regions(include_callable=True)

        assert any(
            "shadowed by a page-level exclusion" in rec.message for rec in caplog.records
        ), "expected a warning when a PDF-level exclusion label is shadowed"
    finally:
        pdf.close()


def test_qa_segment_failures_surface_real_error(monkeypatch):
    service = QAService(context=None)

    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(service, "_ask_single", boom)

    result = service._ask_segments(
        host=object(),
        segments=[object(), object()],
        question="What is the total?",
        min_confidence=0.1,
        model=None,
        debug=False,
    )

    assert result.success is False
    assert "engine exploded" in result.error_message
    assert "2 QA segment(s) failed" in result.error_message


def test_qa_empty_segments_keeps_generic_message():
    service = QAService(context=None)
    result = service._ask_segments(
        host=object(),
        segments=[],
        question="anything",
        min_confidence=0.1,
        model=None,
        debug=False,
    )
    assert result.success is False
    assert "No content available" in result.error_message


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_context_get_service_creates_single_instance_under_threads():
    created = []
    barrier = threading.Barrier(8)

    def slow_factory(context):
        instance = object()
        created.append(instance)
        return instance

    context = PDFContext(service_factories={"slow": slow_factory})
    results = []

    def worker():
        barrier.wait()
        results.append(context.get_service("slow"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == 1, f"factory ran {len(created)} times under contention"
    assert all(r is results[0] for r in results)


# The DelegateRegistry thread-safety tests were removed along with the
# @register_delegate machinery: the registry was write-only (never read)
# since its only consumer was removed on 2025-11-18.


def test_engine_provider_not_found_lists_available():
    provider = EngineProvider()
    provider._entry_points_loaded = True  # skip entry-point discovery
    provider.register("testcap", "alpha", lambda context, **kw: object())
    provider.register("testcap", "beta", lambda context, **kw: object())

    with pytest.raises(LookupError) as excinfo:
        provider.get("testcap", context=None, name="missing")
    message = str(excinfo.value)
    assert "alpha" in message and "beta" in message

    with pytest.raises(LookupError) as excinfo:
        provider.get("emptycap", context=None, name="missing")
    assert "No engines are registered" in str(excinfo.value)


def test_engine_cache_losing_racer_is_cleaned_up():
    cache = EngineCache(maxsize=4)

    class FakeEngine:
        def __init__(self, name):
            self.name = name
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    winner = FakeEngine("winner")
    loser = FakeEngine("loser")
    key_args = ("eng", ("en",), "cpu", "init")

    def losing_factory():
        # Simulate another thread winning the race while we were building:
        # the key gets populated before our freshly built engine is stored.
        cache.get_or_create(*key_args, factory=lambda: winner)
        return loser

    result = cache.get_or_create(*key_args, factory=losing_factory)

    assert result is winner
    assert loser.cleaned, "engine that lost the creation race should be cleaned up"
    assert not winner.cleaned
