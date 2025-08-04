"""
Test element addition functionality for natural-pdf.
"""

from unittest.mock import MagicMock

import pytest

from natural_pdf.elements.base import Element
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


class TestElementAddition:
    """Test adding elements together with + operator."""

    def create_mock_element(self, bbox=(0, 0, 100, 100), element_type="text"):
        """Create a mock element for testing."""
        page = MagicMock()
        obj = {
            "x0": bbox[0],
            "top": bbox[1],
            "x1": bbox[2],
            "bottom": bbox[3],
            "object_type": element_type,
        }
        return Element(obj, page)

    def create_mock_region(self, bbox=(0, 0, 100, 100)):
        """Create a mock region for testing."""
        page = MagicMock()
        return Region(page, bbox)

    def test_element_plus_element(self):
        """Test adding two elements together."""
        elem1 = self.create_mock_element((0, 0, 50, 50))
        elem2 = self.create_mock_element((50, 0, 100, 50))

        result = elem1 + elem2

        assert isinstance(result, ElementCollection)
        assert len(result) == 2
        assert elem1 in result
        assert elem2 in result

    def test_element_plus_region(self):
        """Test adding element and region together."""
        elem = self.create_mock_element((0, 0, 50, 50))
        region = self.create_mock_region((50, 0, 100, 50))

        result = elem + region

        assert isinstance(result, ElementCollection)
        assert len(result) == 2
        assert elem in result
        assert region in result

    def test_region_plus_region(self):
        """Test adding two regions together."""
        region1 = self.create_mock_region((0, 0, 50, 50))
        region2 = self.create_mock_region((50, 0, 100, 50))

        result = region1 + region2

        assert isinstance(result, ElementCollection)
        assert len(result) == 2
        assert region1 in result
        assert region2 in result

    def test_element_plus_collection(self):
        """Test adding element to existing collection."""
        elem1 = self.create_mock_element((0, 0, 50, 50))
        elem2 = self.create_mock_element((50, 0, 100, 50))
        elem3 = self.create_mock_element((100, 0, 150, 50))

        collection = ElementCollection([elem1, elem2])
        result = elem3 + collection

        assert isinstance(result, ElementCollection)
        assert len(result) == 3
        assert all(elem in result for elem in [elem1, elem2, elem3])

    def test_collection_plus_element(self):
        """Test adding collection to element."""
        elem1 = self.create_mock_element((0, 0, 50, 50))
        elem2 = self.create_mock_element((50, 0, 100, 50))
        elem3 = self.create_mock_element((100, 0, 150, 50))

        collection = ElementCollection([elem1, elem2])
        result = collection + elem3

        assert isinstance(result, ElementCollection)
        assert len(result) == 3
        assert all(elem in result for elem in [elem1, elem2, elem3])

    def test_collection_plus_collection(self):
        """Test adding two collections together."""
        elem1 = self.create_mock_element((0, 0, 50, 50))
        elem2 = self.create_mock_element((50, 0, 100, 50))
        elem3 = self.create_mock_element((100, 0, 150, 50))
        elem4 = self.create_mock_element((150, 0, 200, 50))

        collection1 = ElementCollection([elem1, elem2])
        collection2 = ElementCollection([elem3, elem4])
        result = collection1 + collection2

        assert isinstance(result, ElementCollection)
        assert len(result) == 4
        assert all(elem in result for elem in [elem1, elem2, elem3, elem4])

    def test_chained_addition(self):
        """Test chaining multiple additions."""
        elem1 = self.create_mock_element((0, 0, 50, 50))
        elem2 = self.create_mock_element((50, 0, 100, 50))
        elem3 = self.create_mock_element((100, 0, 150, 50))
        elem4 = self.create_mock_element((150, 0, 200, 50))

        result = elem1 + elem2 + elem3 + elem4

        assert isinstance(result, ElementCollection)
        assert len(result) == 4
        assert all(elem in result for elem in [elem1, elem2, elem3, elem4])

    def test_sum_elements(self):
        """Test using sum() on a list of elements."""
        elements = [self.create_mock_element((i * 50, 0, (i + 1) * 50, 50)) for i in range(4)]

        result = sum(elements)

        assert isinstance(result, ElementCollection)
        assert len(result) == 4
        assert all(elem in result for elem in elements)

    def test_add_invalid_type(self):
        """Test that adding invalid type raises TypeError."""
        elem = self.create_mock_element()

        with pytest.raises(TypeError, match="Cannot add Element with"):
            result = elem + "invalid"

        with pytest.raises(TypeError, match="Cannot add Element with"):
            result = elem + 123

        with pytest.raises(TypeError, match="Cannot add Element with"):
            result = elem + None

    def test_real_world_example(self):
        """Test a real-world-like example with regions."""
        # Mock a section
        section = MagicMock()

        # Create mock regions as if they came from find operations
        complainant_region = self.create_mock_region((100, 100, 200, 120))
        dob_region = self.create_mock_region((100, 120, 200, 140))
        gender_region = self.create_mock_region((100, 140, 200, 160))
        phone_region = self.create_mock_region((100, 160, 200, 180))

        # Simulate the user's example
        combined = complainant_region + dob_region + gender_region + phone_region

        assert isinstance(combined, ElementCollection)
        assert len(combined) == 4
        assert all(
            region in combined
            for region in [complainant_region, dob_region, gender_region, phone_region]
        )

        # Should be able to iterate over the collection
        regions_list = list(combined)
        assert len(regions_list) == 4

        # Should be able to access by index
        assert combined[0] == complainant_region
        assert combined[1] == dob_region
        assert combined[2] == gender_region
        assert combined[3] == phone_region
