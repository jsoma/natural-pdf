import pytest


@pytest.fixture
def _sample_region(practice_pdf):
    page = practice_pdf.pages[0]
    height = min(200, page.height)
    return page.region(0, 0, page.width, height)


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"preserve_whitespace": True},
        {"keep_blank_chars": True},
        {"use_exclusions": False},
        {"debug": True},
        {"debug_exclusions": True},
        {"return_textmap": True},
    ],
)
def test_region_extract_text_rejects_legacy_kwargs(_sample_region, legacy_kwargs):
    with pytest.raises(TypeError):
        _sample_region.extract_text(**legacy_kwargs)


def test_region_apply_exclusions_is_the_only_exclusion_switch(monkeypatch, _sample_region):
    call_count = {"value": 0}

    def fake_get_exclusions(self, include_callable=True, debug=False):
        call_count["value"] += 1
        return []

    monkeypatch.setattr(type(_sample_region), "_get_exclusion_regions", fake_get_exclusions)

    _sample_region.extract_text(apply_exclusions=True)
    assert call_count["value"] >= 1

    prior = call_count["value"]
    _sample_region.extract_text(apply_exclusions=False)
    assert call_count["value"] == prior
