from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


def _assert_same_identities(actual, expected) -> None:
    assert [id(region) for region in actual] == [id(region) for region in expected]


def test_same_name_replacement_is_atomic_and_selector_visible(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    first = page.create_region(0, 0, 50, 20)
    unnamed = page.create_region(0, 30, 50, 50)
    replacement = page.create_region(0, 60, 50, 80)

    page.add_region(first, name="header")
    page.add_region(unnamed, source="detected")
    page.add_region(replacement, name="header")

    # Replacement keeps the original named slot, removes the stale identity,
    # and is immediately reflected in both selectors and the compatibility view.
    _assert_same_identities(page.iter_regions(), [replacement, unnamed])
    selected = page.find_all("region[name=header]")
    _assert_same_identities(selected, [replacement])
    assert page._regions["named"] == {"header": replacement}
    _assert_same_identities(page._regions["detected"], [unnamed])


def test_reregistering_named_region_as_unnamed_clears_public_name(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, 50, 20)

    page.add_region(region, name="header")
    page.add_region(region)

    assert region.name is None
    assert page._regions["named"] == {}
    _assert_same_identities(page._regions["detected"], [region])
    assert len(page.find_all("region[name=header]")) == 0
    _assert_same_identities(page.iter_regions(), [region])


def test_region_removal_by_name_and_source_survives_reload(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    named = page.create_region(0, 0, 50, 20)
    detected = page.create_region(0, 30, 50, 50)
    retained = page.create_region(0, 60, 50, 80)

    page.add_region(named, name="header")
    page.add_region(detected, source="detected")
    page.add_region(retained, source="manual")

    assert page.remove_regions(name="header") == 1
    assert page.remove_regions_by_source("detected") == 1
    _assert_same_identities(page.iter_regions(), [retained])

    page.invalidate_element_cache()

    _assert_same_identities(page.iter_regions(), [retained])
    assert len(page.find_all("region[name=header]")) == 0
    assert len(page.find_all("region[source=detected]")) == 0
    assert page._regions["named"] == {}


def test_region_kinds_preserve_identity_order_across_invalidation(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    named = page.create_region(0, 0, 40, 20)
    detected = page.create_region(0, 30, 40, 50)
    checkbox = page.create_region(0, 60, 20, 80)
    ocr_table = page.create_region(0, 90, 100, 140)

    detected.region_type = "title"
    checkbox.region_type = "checkbox"
    ocr_table.region_type = "table"

    page.add_region(named, name="header")
    page.add_region(detected, source="detected")
    page.add_region(checkbox, source="checkbox")
    page.add_region(ocr_table, source="vlm-ocr")
    expected = [named, detected, checkbox, ocr_table]

    _assert_same_identities(page.iter_regions(), expected)
    page.invalidate_element_cache()
    _assert_same_identities(page.iter_regions(), expected)
    _assert_same_identities(page.find_all("region"), expected)

    assert page._regions["named"] == {"header": named}
    _assert_same_identities(page._regions["detected"], expected[1:])

    # Compatibility containers are snapshots, not a second mutable authority.
    page._regions["named"].clear()
    page._regions["detected"].clear()
    _assert_same_identities(page.iter_regions(), expected)


def test_remove_element_region_cannot_resurrect_after_invalidation(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, 50, 20)
    page.add_region(region, name="header")

    assert page.remove_element(region, "regions") is True
    page.invalidate_element_cache()

    assert page.iter_regions() == []
    assert page._regions["named"] == {}


def test_concurrent_same_name_registration_never_leaves_duplicates(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    sentinel = page.create_region(0, 0, 10, 10)
    candidates = [page.create_region(i, 20, i + 1, 30) for i in range(16)]
    page.add_region(sentinel, source="manual")

    def register(region):
        page.add_region(region, name="header")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(register, candidates))

    registered = page.iter_regions()
    assert registered[0] is sentinel
    assert len(registered) == 2
    assert registered[1] in candidates
    _assert_same_identities(page.find_all("region[name=header]"), [registered[1]])


@pytest.mark.parametrize("mutation", ["register", "remove"])
def test_region_mutation_holds_store_transaction_across_load(
    practice_pdf_fresh, monkeypatch, mutation
):
    """Invalidation cannot enter between loading and a Region mutation."""
    page = practice_pdf_fresh.pages[0]
    manager = page._element_mgr
    region = page.create_region(0, 0, 50, 20)
    if mutation == "remove":
        page.add_region(region, name="header")

    loaded = threading.Event()
    release = threading.Event()
    errors = []
    real_load_elements = manager.load_elements

    def paused_load_elements():
        real_load_elements()
        if threading.current_thread().name == "region-mutator":
            loaded.set()
            if not release.wait(timeout=2):
                raise AssertionError("timed out waiting to finish Region mutation")

    monkeypatch.setattr(manager, "load_elements", paused_load_elements)

    def mutate():
        try:
            if mutation == "register":
                page.add_region(region, name="header")
            else:
                page.remove_regions(name="header")
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    worker = threading.Thread(target=mutate, name="region-mutator", daemon=True)
    worker.start()
    assert loaded.wait(timeout=2)

    # If the manager released its transaction after load_elements(), an
    # invalidator could acquire this lock and create a regions-only loaded
    # store before registration/removal completes.
    acquired_between_load_and_mutation = manager._store._lock.acquire(blocking=False)
    if acquired_between_load_and_mutation:
        manager._store._lock.release()
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert not acquired_between_load_and_mutation


@pytest.mark.parametrize(
    ("source", "expected_source"),
    [(None, "named"), ("manual", "manual")],
)
def test_page_region_metadata_is_assigned_inside_manager_registration(
    practice_pdf_fresh, monkeypatch, source, expected_source
):
    """Public selector metadata does not move ahead of its repository key."""
    page = practice_pdf_fresh.pages[0]
    manager = page._element_mgr
    region = page.create_region(0, 0, 50, 20)
    manager_entered = threading.Event()
    release = threading.Event()
    errors = []
    real_add_region = manager.add_region

    def paused_add_region(*args, **kwargs):
        manager_entered.set()
        if not release.wait(timeout=2):
            raise AssertionError("timed out waiting to register Region")
        return real_add_region(*args, **kwargs)

    monkeypatch.setattr(manager, "add_region", paused_add_region)

    def register():
        try:
            page.add_region(region, name="header", source=source)
        except BaseException as error:  # pragma: no branch - asserted below
            errors.append(error)

    worker = threading.Thread(target=register, daemon=True)
    worker.start()
    assert manager_entered.wait(timeout=2)

    # Page delegates metadata ownership without making it observable before
    # the manager has entered its registration transaction.
    assert region.name is None
    assert region.source is None
    assert manager.region_registry_name(region) is None

    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
    assert region.name == "header"
    assert region.source == expected_source
    assert manager.region_registry_name(region) == "header"
    _assert_same_identities(page.find_all("region[name=header]"), [region])
