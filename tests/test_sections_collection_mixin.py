from __future__ import annotations

from typing import Iterable, Optional, Union

from natural_pdf.collections.mixins import SectionsCollectionMixin
from natural_pdf.core.interfaces import SupportsSections
from natural_pdf.elements.element_collection import ElementCollection


class FakeElement:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:  # pragma: no cover - simple passthrough
        return self.text


class FakeSection(SupportsSections):
    def __init__(self, name: str, elements: Iterable[FakeElement]) -> None:
        self.name = name
        self._elements = list(elements)

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Iterable[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: Optional[Union[bool, dict]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
    ) -> ElementCollection:
        # ignore selector args for test purposes
        return ElementCollection(list(self._elements))

    def find(self, *args, **kwargs):  # pragma: no cover - not needed in tests
        raise NotImplementedError

    def get_sections(self, *args, **kwargs):  # pragma: no cover - not needed
        raise NotImplementedError

    def to_region(self):  # pragma: no cover - not needed
        raise NotImplementedError

    def extract_text(self, *args, **kwargs) -> str:
        return " ".join(elem.extract_text() for elem in self._elements)


class FakeCollection(SectionsCollectionMixin):
    def __init__(self, sections: Iterable[FakeSection]):
        self._sections = list(sections)

    def _iter_sections(self) -> Iterable[FakeSection]:
        return iter(self._sections)


def test_find_all_deduplicates_elements() -> None:
    shared_element = FakeElement("shared")
    sections = [
        FakeSection("a", [shared_element, FakeElement("one")]),
        FakeSection("b", [shared_element, FakeElement("two")]),
    ]
    collection = FakeCollection(sections)

    result = collection.find_all("text")
    assert [elem.text for elem in result.elements] == ["shared", "one", "two"]


def test_extract_text_concatenates_each_section() -> None:
    sections = [
        FakeSection("a", [FakeElement("alpha")]),
        FakeSection("b", [FakeElement("beta")]),
    ]
    collection = FakeCollection(sections)

    assert collection.extract_text(separator="|") == "alpha|beta"
