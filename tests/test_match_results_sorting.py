"""Test MatchResults sorting and top() method"""

import pytest
from PIL import Image, ImageDraw

import natural_pdf as npdf


def create_test_pdf_with_varying_logos():
    """Create a test PDF with logos of varying quality/clarity"""
    img = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(img)

    # Logo 1: Perfect match (black on white) at (100, 100)
    draw.rectangle([100, 100, 150, 150], fill="black")
    draw.ellipse([110, 110, 140, 140], fill="white")

    # Logo 2: Good match (dark gray on white) at (300, 100)
    draw.rectangle([300, 100, 350, 150], fill=(50, 50, 50))
    draw.ellipse([310, 110, 340, 140], fill="white")

    # Logo 3: Fair match (with noise) at (100, 300)
    draw.rectangle([100, 300, 150, 350], fill="black")
    draw.ellipse([110, 310, 140, 340], fill="white")
    # Add some noise
    for i in range(100, 150, 5):
        for j in range(300, 350, 5):
            if i % 10 == 0:
                draw.point((i, j), fill=(128, 128, 128))

    # Logo 4: Poor match (faded) at (300, 300)
    draw.rectangle([300, 300, 350, 350], fill=(100, 100, 100))
    draw.ellipse([310, 310, 340, 340], fill=(200, 200, 200))

    # Logo 5: Another good match at (100, 500)
    draw.rectangle([100, 500, 150, 550], fill="black")
    draw.ellipse([110, 510, 140, 540], fill="white")

    return img


def test_match_results_automatic_sorting(tmp_path):
    """Test that MatchResults automatically sorts by confidence"""
    # Create test PDF
    img = create_test_pdf_with_varying_logos()
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load and search
    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    # Use the perfect logo as template
    template = page.region(100, 100, 150, 150)

    # Find all similar logos
    matches = page.find_similar(
        template,
        confidence=0.5,  # Low threshold to get multiple matches
        method="template",
        show_progress=False,
    )

    # Verify matches are sorted by confidence
    assert len(matches) >= 3, f"Expected at least 3 matches, got {len(matches)}"

    # Check that confidences are in descending order
    confidences = [m.confidence for m in matches]
    assert confidences == sorted(
        confidences, reverse=True
    ), f"Matches not sorted by confidence: {confidences}"

    # First match should have highest confidence
    assert matches[0].confidence == max(confidences)

    # Last match should have lowest confidence
    assert matches[-1].confidence == min(confidences)


def test_top_method(tmp_path):
    """Test the top() method returns highest confidence matches"""
    # Create test PDF
    img = create_test_pdf_with_varying_logos()
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load and search
    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    template = page.region(100, 100, 150, 150)
    matches = page.find_similar(template, confidence=0.5, method="template", show_progress=False)

    # Get top 3
    top_3 = matches.top(3)

    # Should return exactly 3 matches
    assert len(top_3) == 3

    # Should be the same as first 3 matches
    assert [m.confidence for m in top_3] == [m.confidence for m in matches[:3]]

    # Top 3 should also be sorted
    confidences = [m.confidence for m in top_3]
    assert confidences == sorted(confidences, reverse=True)

    # Test edge cases
    top_100 = matches.top(100)  # More than available
    assert len(top_100) == len(matches)

    top_0 = matches.top(0)
    assert len(top_0) == 0


def test_iteration_order(tmp_path):
    """Test that iteration yields matches in confidence order"""
    # Create test PDF
    img = create_test_pdf_with_varying_logos()
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load and search
    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    template = page.region(100, 100, 150, 150)
    matches = page.find_similar(template, confidence=0.5, method="template", show_progress=False)

    # Iterate and check order
    prev_confidence = 1.0
    for match in matches:
        assert (
            match.confidence <= prev_confidence
        ), f"Match confidence {match.confidence} > previous {prev_confidence}"
        prev_confidence = match.confidence


def test_filter_preserves_order(tmp_path):
    """Test that filtering preserves confidence order"""
    # Create test PDF
    img = create_test_pdf_with_varying_logos()
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load and search
    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    template = page.region(100, 100, 150, 150)
    matches = page.find_similar(template, confidence=0.5, method="template", show_progress=False)

    # Filter by confidence
    high_conf = matches.filter_by_confidence(0.8)

    # Should still be sorted
    confidences = [m.confidence for m in high_conf]
    assert confidences == sorted(confidences, reverse=True)

    # All should be above threshold
    assert all(m.confidence >= 0.8 for m in high_conf)


def test_combining_top_and_filter(tmp_path):
    """Test combining top() with other operations"""
    # Create test PDF
    img = create_test_pdf_with_varying_logos()
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load and search
    pdf = npdf.PDF(str(pdf_path))
    page = pdf.pages[0]

    template = page.region(100, 100, 150, 150)
    matches = page.find_similar(template, confidence=0.5, method="template", show_progress=False)

    # Get top 5, then filter
    top_5_high_conf = matches.top(5).filter_by_confidence(0.8)

    # Should have at most 5 matches
    assert len(top_5_high_conf) <= 5

    # All should be high confidence
    assert all(m.confidence >= 0.8 for m in top_5_high_conf)

    # Should still be sorted
    confidences = [m.confidence for m in top_5_high_conf]
    assert confidences == sorted(confidences, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
