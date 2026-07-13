"""Concurrency contracts for parent-owned lazy Page materialization."""

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import natural_pdf as npdf
from natural_pdf.core.pdf import PDF, _LazyPageList
from natural_pdf.elements.region import Region


class _SlowPage:
    created = 0
    creation_lock = threading.Lock()

    def __init__(
        self,
        plumber_page,
        *,
        parent,
        index,
        font_attrs=None,
        load_text=True,
        context=None,
    ):
        with type(self).creation_lock:
            type(self).created += 1
        time.sleep(0.05)
        self.plumber_page = plumber_page
        self.parent = parent
        self.index = index
        self.number = index + 1


def _parent_and_pages(page_count=2):
    parent = SimpleNamespace(
        _closed=False,
        _pdf=object(),
        _context=object(),
        _regions=[],
    )
    plumber_pdf = SimpleNamespace(pages=[Mock() for _ in range(page_count)])
    pages = _LazyPageList(parent, plumber_pdf)
    parent._pages = pages
    return parent, pages


def test_concurrent_main_and_slice_access_materializes_one_canonical_page():
    """All views of one actual index must share a single parent-owned Page."""
    parent, pages = _parent_and_pages()
    slices = [pages[:1] for _ in range(8)]
    _SlowPage.created = 0

    with patch("natural_pdf.core.page.Page", _SlowPage):
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(lambda: pages[0]) for _ in range(4)]
            futures.extend(executor.submit(lambda view=view: view[0]) for view in slices)
            resolved = [future.result(timeout=2) for future in futures]

    assert _SlowPage.created == 1
    assert all(page is resolved[0] for page in resolved)
    assert parent._pages._cache[0] is resolved[0]
    assert all(view._cache[0] is resolved[0] for view in slices)


def test_direct_lazy_list_without_parent_pages_reuses_its_local_cache():
    """Compatibility parents without ``_pages`` still materialize only once."""
    parent = SimpleNamespace(
        _closed=False,
        _pdf=object(),
        _context=object(),
        _regions=[],
    )
    plumber_pdf = SimpleNamespace(pages=[Mock()])
    pages = _LazyPageList(parent, plumber_pdf)
    _SlowPage.created = 0

    with patch("natural_pdf.core.page.Page", _SlowPage):
        first = pages[0]
        second = pages[0]

    assert first is second
    assert pages._cache[0] is first
    assert _SlowPage.created == 1


def test_named_region_factory_can_reenter_page_without_deadlock():
    """A region factory can resolve its in-progress parent Page canonically."""
    parent, pages = _parent_and_pages(page_count=1)
    seen = []

    def reentrant_region(page):
        seen.append(parent._pages[0])
        return None

    parent._regions = [(reentrant_region, "header")]
    result = []

    with patch("natural_pdf.core.page.Page", _SlowPage):
        access_thread = threading.Thread(target=lambda: result.append(pages[0]), daemon=True)
        access_thread.start()
        access_thread.join(timeout=2)

    assert not access_thread.is_alive(), "re-entrant Page access deadlocked"
    assert len(result) == 1
    assert seen == [result[0]]
    assert parent._pages._cache[0] is result[0]


def test_cross_page_named_region_factories_cannot_form_lock_cycle():
    """Factories for two Pages can request each other without deadlocking."""
    parent, pages = _parent_and_pages(page_count=2)
    first_factory_entered = threading.Event()
    second_access_started = threading.Event()
    second_page_constructed = threading.Event()
    seen = {}
    errors = []

    class _CrossPage:
        def __init__(
            self,
            plumber_page,
            *,
            parent,
            index,
            font_attrs=None,
            load_text=True,
            context=None,
        ):
            self.plumber_page = plumber_page
            self.parent = parent
            self.index = index
            self.number = index + 1
            if index == 1:
                second_page_constructed.set()

    def cross_page_factory(page):
        if page.index == 0:
            first_factory_entered.set()
            assert second_access_started.wait(timeout=2)
            # Under the former per-page locks, the other thread can construct
            # Page 1 and then block requesting Page 0. Under the shared lock it
            # cannot enter, so this thread constructs Page 1 re-entrantly.
            second_page_constructed.wait(timeout=0.1)
            seen[0] = parent._pages[1]
        else:
            seen[1] = parent._pages[0]
        return None

    parent._regions = [(cross_page_factory, "cross-page")]

    def resolve(index):
        try:
            if index == 1:
                second_access_started.set()
            seen[f"result-{index}"] = pages[index]
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    with patch("natural_pdf.core.page.Page", _CrossPage):
        first = threading.Thread(target=resolve, args=(0,), daemon=True)
        second = threading.Thread(target=resolve, args=(1,), daemon=True)
        first.start()
        assert first_factory_entered.wait(timeout=2)
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive(), "Page 0 factory deadlocked requesting Page 1"
    assert not second.is_alive(), "Page 1 factory deadlocked requesting Page 0"
    assert errors == []
    assert seen[0] is seen["result-1"]
    assert seen[1] is seen["result-0"]
    assert parent._pages._cache == [seen["result-0"], seen["result-1"]]


def test_pdf_add_region_cannot_miss_page_published_concurrently():
    """Factory append/application and Page publication have one ordering."""
    publication_started = threading.Event()
    release_publication = threading.Event()
    registry_lock_attempted = threading.Event()
    errors = []
    factory_pages = []

    class _BlockingCache(list):
        def __setitem__(self, index, value):
            if value is not None:
                publication_started.set()
                assert release_publication.wait(timeout=2)
            super().__setitem__(index, value)

    class _FactoryPage:
        def __init__(
            self,
            plumber_page,
            *,
            parent,
            index,
            font_attrs=None,
            load_text=True,
            context=None,
        ):
            self.plumber_page = plumber_page
            self.parent = parent
            self.index = index
            self.number = index + 1
            self.added_regions = []

        def add_region(self, region, name=None, *, source=None):
            self.added_regions.append((region, name, source))
            return self

    parent, pages = _parent_and_pages(page_count=1)
    pages._cache = _BlockingCache(pages._cache)
    pages._materialization_state()
    materialization_lock = parent._page_materialization_state.registry_lock()

    class _TrackedRegistryLock:
        def __enter__(self):
            registry_lock_attempted.set()
            materialization_lock.acquire()

        def __exit__(self, exc_type, exc_value, traceback):
            materialization_lock.release()

    parent._page_materialization_state.registry_lock = lambda: _TrackedRegistryLock()

    # Mock(spec=Region) passes the production isinstance check without needing
    # a fully-fledged natural-pdf Page in this low-level publication test.
    produced_region = Mock(spec=Region)

    def factory(page):
        factory_pages.append(page)
        return produced_region

    def materialize():
        try:
            pages[0]
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    def register():
        try:
            PDF.add_region(parent, factory, name="header")
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    with patch("natural_pdf.core.page.Page", _FactoryPage):
        page_thread = threading.Thread(target=materialize, daemon=True)
        page_thread.start()
        assert publication_started.wait(timeout=2)

        region_thread = threading.Thread(target=register, daemon=True)
        region_thread.start()
        assert registry_lock_attempted.wait(timeout=2)
        assert factory_pages == []

        release_publication.set()
        page_thread.join(timeout=2)
        region_thread.join(timeout=2)

    assert not page_thread.is_alive()
    assert not region_thread.is_alive()
    assert errors == []
    published = pages._cache[0]
    assert factory_pages == [published]
    assert published.added_regions == [(produced_region, "header", "named")]


def _two_page_pdf_bytes() -> bytes:
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000186 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
257
%%EOF"""


def test_close_waits_for_materialization_and_prevents_late_publication():
    """Closing while a Page is loading cannot return or cache a half-raced Page."""
    started = threading.Event()
    release = threading.Event()

    class _BlockingPage:
        def __init__(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=2)
            self.number = 1

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as stream:
        stream.write(_two_page_pdf_bytes())
        pdf_path = stream.name

    pdf = npdf.PDF(pdf_path)
    try:
        with patch("natural_pdf.core.page.Page", _BlockingPage):
            access_result = []

            def access_page():
                try:
                    access_result.append(pdf.pages[0])
                except Exception as error:  # pragma: no branch - assertion below
                    access_result.append(error)

            access_thread = threading.Thread(target=access_page)
            access_thread.start()
            assert started.wait(timeout=2)

            close_thread = threading.Thread(target=pdf.close)
            close_thread.start()
            for _ in range(100):
                if pdf._page_materialization_state.is_closing():
                    break
                time.sleep(0.01)
            assert pdf._page_materialization_state.is_closing()
            assert close_thread.is_alive(), "close should wait for the active page read"

            release.set()
            access_thread.join(timeout=2)
            close_thread.join(timeout=2)

        assert not access_thread.is_alive()
        assert not close_thread.is_alive()
        assert len(access_result) == 1
        assert isinstance(access_result[0], RuntimeError)
        assert pdf._pages._cache[0] is None
        with pytest.raises(RuntimeError, match="parent PDF has been closed"):
            pdf.pages[0]
    finally:
        pdf.close()
        Path(pdf_path).unlink(missing_ok=True)


def test_publication_and_close_have_one_atomic_ordering():
    """Close cannot slip between the final open check and cache publication."""
    publication_started = threading.Event()
    release_publication = threading.Event()

    class _BlockingCache(list):
        def __setitem__(self, index, value):
            if value is not None:
                publication_started.set()
                assert release_publication.wait(timeout=2)
            super().__setitem__(index, value)

    class _Page:
        def __init__(self, *args, **kwargs):
            self.number = 1

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as stream:
        stream.write(_two_page_pdf_bytes())
        pdf_path = stream.name

    pdf = npdf.PDF(pdf_path)
    try:
        pdf._pages._cache = _BlockingCache(pdf._pages._cache)
        access_result = []
        with patch("natural_pdf.core.page.Page", _Page):
            access_thread = threading.Thread(target=lambda: access_result.append(pdf.pages[0]))
            access_thread.start()
            assert publication_started.wait(timeout=2)

            close_thread = threading.Thread(target=pdf.close)
            close_thread.start()
            time.sleep(0.05)

            assert close_thread.is_alive(), "close crossed an in-progress cache publication"
            assert not pdf._closed

            release_publication.set()
            access_thread.join(timeout=2)
            close_thread.join(timeout=2)

        assert not access_thread.is_alive()
        assert not close_thread.is_alive()
        assert len(access_result) == 1
        assert pdf._pages._cache[0] is access_result[0]
    finally:
        release_publication.set()
        pdf.close()
        Path(pdf_path).unlink(missing_ok=True)
