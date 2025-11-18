from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF
from natural_pdf.elements.base import Element
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion
from natural_pdf.services.methods import (
    classification_methods,
    navigation_methods,
    ocr_methods,
    text_methods,
)


def _assert_delegate(cls, method_name, helper):
    method = getattr(cls, method_name)
    wrapped = getattr(method, "__wrapped__", None)
    assert (
        wrapped is helper
    ), f"{cls.__name__}.{method_name} should wrap {helper.__module__}.{helper.__name__}"


def test_text_delegates_use_helpers():
    _assert_delegate(Region, "update_text", text_methods.update_text)
    _assert_delegate(Region, "correct_ocr", text_methods.correct_ocr)
    _assert_delegate(Element, "update_text", text_methods.update_text)
    _assert_delegate(Element, "correct_ocr", text_methods.correct_ocr)
    _assert_delegate(PageCollection, "update_text", text_methods.update_text)


def test_ocr_delegates_use_helpers():
    _assert_delegate(Page, "apply_ocr", ocr_methods.apply_ocr)
    _assert_delegate(Page, "extract_ocr_elements", ocr_methods.extract_ocr_elements)
    _assert_delegate(Page, "apply_custom_ocr", ocr_methods.apply_custom_ocr)
    _assert_delegate(Page, "remove_ocr_elements", ocr_methods.remove_ocr_elements)
    _assert_delegate(Page, "clear_text_layer", ocr_methods.clear_text_layer)
    _assert_delegate(
        Page, "create_text_elements_from_ocr", ocr_methods.create_text_elements_from_ocr
    )
    _assert_delegate(PageCollection, "apply_ocr", ocr_methods.apply_ocr)


def test_classification_delegates_use_helpers():
    _assert_delegate(Page, "classify", classification_methods.classify)
    _assert_delegate(Region, "classify", classification_methods.classify)
    _assert_delegate(Element, "classify", classification_methods.classify)
    _assert_delegate(PDF, "classify", classification_methods.classify)


def test_navigation_delegates_use_helpers():
    _assert_delegate(FlowRegion, "above", navigation_methods.above)
    _assert_delegate(FlowRegion, "below", navigation_methods.below)
    _assert_delegate(FlowRegion, "left", navigation_methods.left)
    _assert_delegate(FlowRegion, "right", navigation_methods.right)


def test_register_capability_applies_to_hosts_and_subclasses(monkeypatch):
    from natural_pdf.services import delegates
    from natural_pdf.services.base import ServiceHostMixin

    saved_entries = list(delegates._REGISTERED_CUSTOM_CAPABILITIES)
    try:

        class BaseHost(ServiceHostMixin):
            def extract_text(self):
                return "BASE"

        # Limit registration to this host to avoid mutating real classes.
        monkeypatch.setattr(delegates, "_iter_service_hosts", lambda: [BaseHost])

        def summarize(self, prefix=""):
            return prefix + self.extract_text()

        delegates.register_capability("unit_summary", summarize)

        assert hasattr(BaseHost, "summarize")
        assert BaseHost().summarize(prefix=">") == ">BASE"

        class ChildHost(BaseHost):
            def extract_text(self):
                return "CHILD"

        assert hasattr(ChildHost, "summarize")
        assert ChildHost().summarize(prefix="<") == "<CHILD"
    finally:
        delegates._REGISTERED_CUSTOM_CAPABILITIES[:] = saved_entries
