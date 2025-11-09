#!/usr/bin/env python3
"""Simple test for guides from_content apply_exclusions parameter."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides

TEST_PDF = Path(__file__).parent.parent / "pdfs/01-practice.pdf"


def test_from_content_apply_exclusions_parameter():
    """Test that from_content accepts and uses apply_exclusions parameter."""

    pdf = PDF(TEST_PDF)
    page = pdf[0]

    # Test that the parameter is accepted and works
    guides1 = Guides.from_content(
        obj=page, axis="vertical", markers=["test"], apply_exclusions=True
    )

    guides2 = Guides.from_content(
        obj=page, axis="vertical", markers=["test"], apply_exclusions=False
    )

    # Both should succeed
    assert hasattr(guides1, "vertical")
    assert hasattr(guides2, "vertical")

    # Test instance method too
    guides3 = Guides(page)
    result = guides3.add_content(markers=["test"], apply_exclusions=True)

    assert result is guides3  # Should return self for chaining


def test_apply_exclusions_signature():
    """Test that apply_exclusions parameter has correct signature."""

    from inspect import signature

    # Test class method
    sig = signature(Guides.from_content)
    params = sig.parameters

    assert "apply_exclusions" in params
    assert params["apply_exclusions"].default is True

    # Test instance method
    guides = Guides([100], [100])  # Create with dummy data
    sig2 = signature(guides.add_content)
    params2 = sig2.parameters

    assert "apply_exclusions" in params2
    assert params2["apply_exclusions"].default is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
