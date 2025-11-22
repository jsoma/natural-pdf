import logging

import pytest


@pytest.fixture
def _sample_region(practice_pdf):
    page = practice_pdf.pages[0]
    height = min(200, page.height)
    return page.region(0, 0, page.width, height)


def test_region_extract_text_accepts_legacy_kwargs(_sample_region, caplog):
    caplog.clear()
    caplog.set_level(logging.WARNING, logger="natural_pdf.text.operations")

    text = _sample_region.extract_text(preserve_whitespace=True, use_exclusions=False)
    assert isinstance(text, str)

    offending = [
        record.getMessage()
        for record in caplog.records
        if record.name == "natural_pdf.text.operations"
    ]
    assert all("preserve_whitespace" not in msg for msg in offending)
    assert all("use_exclusions" not in msg for msg in offending)


def test_region_use_exclusions_alias(monkeypatch, _sample_region):
    call_count = {"value": 0}

    def fake_get_exclusions(self, include_callable=True, debug=False):
        call_count["value"] += 1
        return []

    monkeypatch.setattr(type(_sample_region), "_get_exclusion_regions", fake_get_exclusions)

    _sample_region.extract_text()
    assert call_count["value"] >= 1

    prior = call_count["value"]
    _sample_region.extract_text(use_exclusions=False)
    assert call_count["value"] == prior
