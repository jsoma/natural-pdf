"""Test find_similar on regions"""

import pytest
from PIL import Image, ImageDraw

import natural_pdf as npdf


def create_test_pdf_image():
    """Create a test image that simulates a PDF page with logos"""
    img = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(img)

    # Draw some text blocks
    for y in range(100, 700, 100):
        draw.rectangle([50, y, 550, y + 20], fill="black")

    # Draw logo in top-left region (100, 50)
    draw.rectangle([100, 50, 150, 100], fill="blue")
    draw.ellipse([110, 60, 140, 90], fill="white")

    # Draw same logo in middle region (300, 350)
    draw.rectangle([300, 350, 350, 400], fill="blue")
    draw.ellipse([310, 360, 340, 390], fill="white")

    # Draw same logo in bottom-right region (450, 650)
    draw.rectangle([450, 650, 500, 700], fill="blue")
    draw.ellipse([460, 660, 490, 690], fill="white")

    # Draw different shape in another region (100, 450)
    draw.rectangle([100, 450, 150, 500], fill="red")
    draw.rectangle([110, 460, 140, 490], fill="yellow")

    return img


@pytest.fixture
def test_pdf(tmp_path):
    """Create a test PDF with logos"""
    # Create test image
    img = create_test_pdf_image()

    # Save as temporary PDF
    pdf_path = tmp_path / "test_logos.pdf"
    img.save(str(pdf_path), "PDF")

    # Load with natural-pdf
    return npdf.PDF(str(pdf_path))


def test_find_similar_on_region(test_pdf):
    """Test that find_similar works on regions"""
    page = test_pdf.pages[0]

    # Define a region that contains the middle and bottom logos but not the top one
    # Region from y=300 to y=750 (excludes top logo at y=50-100)
    search_region = page.region(0, 300, 600, 750)

    # Get the top logo as our template
    logo_region = page.region(100, 50, 150, 100)

    # Search for similar logos within the search region
    matches = search_region.find_similar(logo_region, confidence=0.8, method="template")

    # Should find 2 matches (middle and bottom logos, not the top one)
    assert len(matches) == 2

    # Check that matches are in the correct locations
    match_centers = [(m.center_x, m.center_y) for m in matches]

    # Middle logo should be around (325, 375)
    assert any(abs(x - 325) < 10 and abs(y - 375) < 10 for x, y in match_centers)

    # Bottom logo should be around (475, 675)
    assert any(abs(x - 475) < 10 and abs(y - 675) < 10 for x, y in match_centers)

    # The top logo (125, 75) should NOT be found since it's outside the search region
    assert not any(y < 300 for _, y in match_centers)


def test_find_similar_on_small_region(test_pdf):
    """Test find_similar on a small region"""
    page = test_pdf.pages[0]

    # Define a small region around the middle logo
    small_region = page.region(250, 300, 400, 450)

    # Get the logo template
    logo_template = page.region(300, 350, 350, 400)

    # Search within the small region
    matches = small_region.find_similar(logo_template, confidence=0.9, method="template")

    # Should find exactly 1 match (the logo itself)
    assert len(matches) == 1
    assert abs(matches[0].center_x - 325) < 5
    assert abs(matches[0].center_y - 375) < 5


def test_find_similar_region_to_region_phash(test_pdf):
    """Test find_similar with perceptual hash method on regions"""
    page = test_pdf.pages[0]

    # Get logo as template
    logo_region = page.region(100, 50, 150, 100)

    # Search in bottom half of page
    bottom_half = page.region(0, 400, 600, 800)

    # Use perceptual hash method
    matches = bottom_half.find_similar(logo_region, confidence=0.8, method="phash")

    # Should find the bottom logo
    assert len(matches) >= 1

    # Check that we found the bottom logo
    found_bottom = any(abs(m.center_x - 475) < 20 and abs(m.center_y - 675) < 20 for m in matches)
    assert found_bottom


def test_find_similar_with_multiple_examples(test_pdf):
    """Test find_similar with multiple example regions"""
    page = test_pdf.pages[0]

    # Get two different logos as examples
    blue_logo = page.region(100, 50, 150, 100)
    red_logo = page.region(100, 450, 150, 500)

    # Search entire page for both
    matches = page.find_similar([blue_logo, red_logo], confidence=0.8, method="template")

    # Should find 3 blue logos and 1 red logo = 4 total
    assert len(matches) >= 4

    # Verify we found all blue logos
    blue_locations = [(125, 75), (325, 375), (475, 675)]
    for expected_x, expected_y in blue_locations:
        found = any(
            abs(m.center_x - expected_x) < 20 and abs(m.center_y - expected_y) < 20 for m in matches
        )
        assert found, f"Missing blue logo at ({expected_x}, {expected_y})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
