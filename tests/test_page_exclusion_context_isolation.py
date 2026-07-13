"""Regression tests for Page exclusion-recursion context isolation."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_without_exclusions_is_nested_and_exception_safe(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    assert page._computing_exclusions is False

    with page.without_exclusions():
        assert page._computing_exclusions is True

        with pytest.raises(RuntimeError, match="expected"):
            with page.without_exclusions():
                assert page._computing_exclusions is True
                raise RuntimeError("expected")

        assert page._computing_exclusions is True

    assert page._computing_exclusions is False


def test_without_exclusions_is_thread_local(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    def observe_context() -> bool:
        return page._computing_exclusions

    with page.without_exclusions():
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(observe_context).result() is False
        assert page._computing_exclusions is True

    assert page._computing_exclusions is False


def test_without_exclusions_is_task_local(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def worker() -> bool:
            with page.without_exclusions():
                entered.set()
                await release.wait()
                return page._computing_exclusions

        task = asyncio.create_task(worker())
        await entered.wait()
        assert page._computing_exclusions is False
        release.set()
        assert await task is True

    asyncio.run(scenario())
    assert page._computing_exclusions is False


def test_legacy_computing_exclusions_assignment_is_preserved(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    page._computing_exclusions = True
    assert page._computing_exclusions is True

    page._computing_exclusions = False
    assert page._computing_exclusions is False
