#!/usr/bin/env python3
"""Simple test for guides from_content apply_exclusions parameter."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def find_test_pdf():
    """Find a test PDF."""
    project_root = Path(__file__).parent.parent
    pdf_path = (
        project_root
        / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
    )
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_from_content_apply_exclusions_parameter():
    """Test that from_content accepts and uses apply_exclusions parameter."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
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
