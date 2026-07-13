"""Regression tests for context-scoped directional navigation defaults."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest

import natural_pdf
from natural_pdf import PDF
from natural_pdf.core.context import PDFContext
from natural_pdf.core.navigation_context import get_directional_within
from natural_pdf.elements.base import DirectionalMixin
from natural_pdf.services.navigation_service import NavigationService


def test_region_within_is_nested_exception_safe_and_does_not_mutate_global_option():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        global_constraint = page.region(left=0, right=10)
        outer = page.region(left=20, right=200)
        inner = page.region(left=40, right=100)
        original = natural_pdf.options.layout.directional_within
        natural_pdf.options.layout.directional_within = global_constraint

        try:
            with outer.within():
                assert get_directional_within() is outer
                assert natural_pdf.options.layout.directional_within is global_constraint

                with pytest.raises(RuntimeError, match="expected"):
                    with inner.within():
                        assert get_directional_within() is inner
                        raise RuntimeError("expected")

                assert get_directional_within() is outer

            assert get_directional_within() is None
            assert natural_pdf.options.layout.directional_within is global_constraint
        finally:
            natural_pdf.options.layout.directional_within = original
    finally:
        pdf.close()


def test_region_within_is_thread_local():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        left = page.region(left=0, right=100)
        right = page.region(left=100, right=200)

        def active_constraint(region):
            with region.within():
                return get_directional_within()

        with ThreadPoolExecutor(max_workers=2) as executor:
            observed = list(executor.map(active_constraint, (left, right)))

        assert observed == [left, right]
        assert get_directional_within() is None
    finally:
        pdf.close()


def test_one_region_context_object_can_be_reused_by_concurrent_tasks():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        region = pdf.pages[0].region(left=0, right=100)
        shared_context = region.within()

        async def scenario():
            both_entered = asyncio.Event()
            entered_count = 0
            entered_lock = asyncio.Lock()

            async def worker():
                nonlocal entered_count
                with shared_context:
                    async with entered_lock:
                        entered_count += 1
                        if entered_count == 2:
                            both_entered.set()
                    await both_entered.wait()
                    assert get_directional_within() is region
                assert get_directional_within() is None

            await asyncio.gather(worker(), worker())

        asyncio.run(scenario())
        assert get_directional_within() is None
    finally:
        pdf.close()


def test_directional_offset_uses_pdf_context_before_global_default():
    context = PDFContext(options={"layout": {"directional_offset": 7.5}})
    pdf = PDF("pdfs/01-practice.pdf", context=context)
    try:
        page = pdf.pages[0]
        source = page.find("text")
        result = source.below(height=10)

        assert result.top == pytest.approx(source.bottom + 7.5)
    finally:
        pdf.close()


@dataclass
class _FakePDF:
    pages: list[Any]


@dataclass
class _FakePage:
    width: float = 100
    height: float = 100
    index: int = 0
    _context: PDFContext | None = None
    pdf: _FakePDF | None = None


class _MultipageProbe(DirectionalMixin):
    """A minimal host that records whether the multipage path was selected."""

    def __init__(self, context: PDFContext):
        self._context = context
        self.page = _FakePage(_context=context)
        self.page.pdf = _FakePDF([self.page])
        self.x0, self.top, self.x1, self.bottom = 10, 10, 20, 20
        self.called = False

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.bottom - self.top

    def _direction_multipage(self, **kwargs):
        self.called = True
        return "context-selected-multipage"


def test_auto_multipage_uses_pdf_context_default():
    host = _MultipageProbe(PDFContext(options={"layout": {"auto_multipage": True}}))

    result = host.below(height=100)

    assert result == "context-selected-multipage"
    assert host.called is True


class _RecordingHost:
    def __init__(self, context: PDFContext):
        self._context = context
        self.kwargs: dict[str, Any] = {}

    def _direction(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


def test_navigation_service_uses_the_hosts_pdf_context_for_offset():
    context = PDFContext(options={"layout": {"directional_offset": 3.25}})
    host = _RecordingHost(context)

    NavigationService(PDFContext.with_defaults()).below(host)

    assert host.kwargs["offset"] == 3.25
