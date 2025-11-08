from __future__ import annotations

from types import MethodType, SimpleNamespace

from natural_pdf.flows.region import FlowRegion


class _OCRStub:
    def __init__(self):
        self.page = SimpleNamespace(_element_mgr=SimpleNamespace())
        self.apply_calls = []
        self.extract_calls = []
        self._exclusion_regions = []

    def apply_ocr(self, *args, **kwargs):
        self.apply_calls.append((args, kwargs))
        return self

    def extract_ocr_elements(self, *args, **kwargs):
        self.extract_calls.append((args, kwargs))
        return [f"elements-{len(self.extract_calls)}"]

    def _get_exclusion_regions(self, *_, **__):
        return list(self._exclusion_regions)


def _flow_region_with_stubs(count: int) -> FlowRegion:
    regions = [_OCRStub() for _ in range(count)]
    flow = SimpleNamespace(arrangement="vertical")
    return FlowRegion(flow=flow, constituent_regions=regions)


def test_flow_region_apply_and_extract_ocr_delegate_to_all_regions():
    flow_region = _flow_region_with_stubs(2)

    flow_region.apply_ocr(engine="easyocr", resolution=150)
    for stub in flow_region.constituent_regions:
        assert len(stub.apply_calls) == 1
        assert stub.apply_calls[0][1] == {"engine": "easyocr", "resolution": 150}

    extracted = flow_region.extract_ocr_elements(engine="surya")
    assert len(extracted) == 2
    for stub in flow_region.constituent_regions:
        assert len(stub.extract_calls) == 1
        assert stub.extract_calls[0][1] == {"engine": "surya"}


def test_flow_region_exclusion_mixin_merges_local_and_child_regions():
    flow_region = _flow_region_with_stubs(2)

    shared_region = object()
    for stub in flow_region.constituent_regions:
        stub._exclusion_regions = [shared_region]

    local_region = object()
    flow_region._exclusions = [("placeholder", None, "region")]

    def fake_evaluator(self, entries, include_callable, debug):
        return [local_region]

    flow_region._evaluate_exclusion_entries = MethodType(fake_evaluator, flow_region)

    regions = flow_region._get_exclusion_regions()
    assert regions[0] is local_region
    assert regions[1] is shared_region
    assert len(regions) == 2  # ensures deduplication
