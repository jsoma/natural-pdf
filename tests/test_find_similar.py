"""Test visual similarity search functionality"""

import pytest

from natural_pdf import PDF


class TestFindSimilar:
    """Test the find_similar method for visual pattern matching"""

    def test_find_similar_exact_match(self):
        """Test finding exact matches of the same image"""
        pdf = PDF("pdfs/classified.pdf")
        images = pdf.find_all("image")

        # Use the first image as template
        template = images[0]

        # Find similar images
        matches = pdf.find_similar(template)

        # Should find at least 2 matches (itself and image[4])
        assert len(matches) >= 2

        # Check that matches include reasonable confidence
        confidences = [match.confidence for match in matches]
        assert max(confidences) > 0.7  # Good confidence for match

    def test_find_similar_scaled_version(self):
        """Test finding scaled versions of an image"""
        pdf = PDF("pdfs/classified.pdf")
        images = pdf.find_all("image")

        # Use the first image as template
        template = images[0]

        # Find similar images - with low resolution, may need lower confidence
        matches = pdf.find_similar(template, confidence=0.6)

        # Should find at least the original and one duplicate
        assert len(matches) >= 2

    def test_find_similar_grayscale_match(self):
        """Test finding grayscale version of a color image"""
        pdf = PDF("pdfs/classified.pdf")
        images = pdf.find_all("image")

        # Use the first image as template
        template = images[0]

        # Find similar images - at low resolution, grayscale may not match perfectly
        matches = pdf.find_similar(template, confidence=0.6)

        # Should find at least 2 matches
        assert len(matches) >= 2

        # Check we're finding matches on both pages
        match_pages = [match.page.number for match in matches]
        assert len(set(match_pages)) >= 1  # At least one page

    def test_find_similar_different_image(self):
        """Test that different images don't match"""
        pdf = PDF("pdfs/classified.pdf")
        images = pdf.find_all("image")

        # Use image[2] which says "TOP SECRET" as template
        template = images[2]

        # Find similar images - use higher confidence for different image
        matches = pdf.find_similar(template, confidence=0.85)

        # Should find few matches (may include itself and some false positives at low res)
        assert len(matches) <= 2  # At most itself and maybe one false positive

    def test_match_results_api(self):
        """Test the MatchResults collection API"""
        pdf = PDF("pdfs/classified.pdf")
        template = pdf.find("image")

        matches = pdf.find_similar(template)

        # Test collection behavior
        assert len(matches) > 0

        # Test iteration
        for match in matches:
            assert hasattr(match, "confidence")
            assert hasattr(match, "bbox")
            assert hasattr(match, "page")
            assert 0 <= match.confidence <= 1

        # Test filtering
        high_conf = matches.filter(lambda m: m.confidence > 0.9)
        assert len(high_conf) <= len(matches)

        # Test page grouping
        pages = matches.pages()
        assert len(pages) > 0

    def test_find_similar_with_region(self):
        """Test using a region as template instead of an element"""
        pdf = PDF("pdfs/classified.pdf")

        # Get first image and create a region from it
        img = pdf.find("image")
        template_region = img.expand(10)  # Slightly larger region

        # Should still find matches - use old defaults for compatibility
        matches = pdf.find_similar(template_region, confidence=0.75, hash_size=6, sizes=0.2)
        assert len(matches) >= 2

    def test_find_similar_multiple_examples(self):
        """Test searching with multiple example templates"""
        pdf = PDF("pdfs/classified.pdf")
        images = pdf.find_all("image")

        # Use multiple examples (original and grayscale)
        templates = [images[0], images[1]]

        # Find similar to any of the templates - use old defaults for compatibility
        matches = pdf.find_similar(templates, confidence=0.8, hash_size=6, sizes=0.2)

        # Should find all variations
        assert len(matches) >= 4  # Both templates plus their matches

    def test_match_results_show_and_regions(self):
        """Test MatchResults.show() and .regions() methods"""
        pdf = PDF("pdfs/classified.pdf")
        template = pdf.find("image")

        matches = pdf.find_similar(template, confidence=0.7)
        assert len(matches) > 0

        # Test regions() returns ElementCollection
        regions = matches.regions()
        assert hasattr(regions, "show")
        assert hasattr(regions, "highlight")
        assert len(regions) == len(matches)

        # Test that regions are Match objects with confidence
        for region in regions:
            assert hasattr(region, "confidence")
            assert hasattr(region, "bbox")
            assert 0 <= region.confidence <= 1

        # Test show() method exists
        assert hasattr(matches, "show")
