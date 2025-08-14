"""Test template matching functionality"""

import numpy as np
import pytest
from PIL import Image

from natural_pdf.vision.similarity import VisualMatcher
from natural_pdf.vision.template_matching import TemplateMatcher


def create_test_image(width=200, height=200):
    """Create a test image with some rectangles"""
    img = Image.new("L", (width, height), color=255)
    pixels = img.load()

    # Scale positions based on image size
    box_size = width // 10
    pos1 = width // 4
    pos2 = 3 * width // 4

    # Draw a black square at scaled position
    for i in range(pos1, pos1 + box_size):
        for j in range(pos1, pos1 + box_size):
            if i < width and j < height:
                pixels[i, j] = 0

    # Draw another black square if space allows
    if pos2 + box_size <= width:
        for i in range(pos2, pos2 + box_size):
            for j in range(pos1, pos1 + box_size):
                if i < width and j < height:
                    pixels[i, j] = 0

    # Draw a different pattern if space allows
    if pos1 + box_size <= width and pos2 + box_size <= height:
        for i in range(pos1, pos1 + box_size):
            for j in range(pos2, pos2 + box_size):
                if i < width and j < height and (i + j) % 2 == 0:
                    pixels[i, j] = 0

    return img


def test_template_matcher_zncc():
    """Test zero-mean normalized cross-correlation"""
    matcher = TemplateMatcher(method="zncc")

    # Create test images
    target = create_test_image()

    # Get actual box positions
    box_size = 20  # 200 // 10
    pos1 = 50  # 200 // 4
    pos2 = 150  # 3 * 200 // 4

    # Extract template with some edge (include border)
    template = target.crop((pos1 - 5, pos1 - 5, pos1 + box_size + 5, pos1 + box_size + 5))

    # Convert to numpy arrays
    target_array = np.array(target, dtype=np.float32) / 255.0
    template_array = np.array(template, dtype=np.float32) / 255.0

    # Run matching
    scores = matcher.match_template(target_array, template_array, step=1)

    # Find the maximum score location
    max_y, max_x = np.unravel_index(np.argmax(scores), scores.shape)
    max_score = scores[max_y, max_x]

    # Should find perfect match
    assert max_score > 0.99, f"Max score {max_score} at ({max_x}, {max_y})"

    # The match should be at the original position (adjusted for border)
    assert abs(max_x - (pos1 - 5)) <= 1 and abs(max_y - (pos1 - 5)) <= 1

    # Should also find good match at the other square location
    if pos2 - 5 < scores.shape[1] - template_array.shape[1]:
        other_score = scores[pos1 - 5, pos2 - 5]
        assert other_score > 0.9, f"Second match score {other_score}"


def test_template_matcher_ncc():
    """Test normalized cross-correlation"""
    matcher = TemplateMatcher(method="ncc")

    target = create_test_image()
    box_size = 20
    pos1 = 50

    # Include edge for variation
    template = target.crop((pos1 - 5, pos1 - 5, pos1 + box_size + 5, pos1 + box_size + 5))

    target_array = np.array(target, dtype=np.float32) / 255.0
    template_array = np.array(template, dtype=np.float32) / 255.0

    scores = matcher.match_template(target_array, template_array, step=1)

    # Should find matches
    max_score = np.max(scores)
    assert max_score > 0.99


def test_template_matcher_ssd():
    """Test sum of squared differences"""
    matcher = TemplateMatcher(method="ssd")

    target = create_test_image()
    box_size = 20
    pos1 = 50

    # Include edge
    template = target.crop((pos1 - 5, pos1 - 5, pos1 + box_size + 5, pos1 + box_size + 5))

    target_array = np.array(target, dtype=np.float32) / 255.0
    template_array = np.array(template, dtype=np.float32) / 255.0

    scores = matcher.match_template(target_array, template_array, step=1)

    # SSD converted to similarity - should find matches
    max_score = np.max(scores)
    assert max_score > 0.99


def test_template_matcher_step():
    """Test template matching with step size"""
    matcher = TemplateMatcher()

    target = create_test_image()
    box_size = 20
    pos1 = 50

    # Include edge
    template = target.crop((pos1 - 5, pos1 - 5, pos1 + box_size + 5, pos1 + box_size + 5))

    target_array = np.array(target, dtype=np.float32) / 255.0
    template_array = np.array(template, dtype=np.float32) / 255.0

    # Step size 5 should be faster but less precise
    scores = matcher.match_template(target_array, template_array, step=5)

    # Should still find approximate matches
    # Note: with step=5, exact position might be slightly off
    max_score_idx = np.unravel_index(np.argmax(scores), scores.shape)
    # Convert from score indices to image coordinates
    actual_y = max_score_idx[0] * 5
    actual_x = max_score_idx[1] * 5

    assert abs(actual_x - (pos1 - 5)) <= 5  # Within 1 step
    assert abs(actual_y - (pos1 - 5)) <= 5


def test_visual_matcher_template_method():
    """Test VisualMatcher with template method"""
    matcher = VisualMatcher()

    # Create test images
    target = create_test_image()
    box_size = 20
    pos1 = 50
    pos2 = 150

    # Include edge for variation
    template = target.crop((pos1 - 5, pos1 - 5, pos1 + box_size + 5, pos1 + box_size + 5))

    # Find matches using template method
    # Note: with step=2, exact alignment is missed, so using lower threshold
    matches = matcher.find_matches_in_image(
        template, target, confidence_threshold=0.8, method="template", step=2
    )

    # Should find at least 2 matches (the two black squares)
    assert len(matches) >= 2

    # Check that matches are at expected locations (adjusted for border)
    match_locs = [(m.bbox[0], m.bbox[1]) for m in matches]

    # Should find a match near pos1-5, pos1-5 (adjusted for border)
    found_first = any(abs(x - (pos1 - 5)) <= 2 and abs(y - (pos1 - 5)) <= 2 for x, y in match_locs)
    assert found_first, f"Expected match near ({pos1-5}, {pos1-5}), got {match_locs}"

    # Should find a match near pos2-5, pos1-5 (adjusted for border)
    found_second = any(abs(x - (pos2 - 5)) <= 2 and abs(y - (pos1 - 5)) <= 2 for x, y in match_locs)
    assert found_second, f"Expected match near ({pos2-5}, {pos1-5}), got {match_locs}"


def test_visual_matcher_multiscale():
    """Test template matching at multiple scales"""
    matcher = VisualMatcher()

    # Create simple test case
    target = Image.new("L", (200, 200), color=255)
    pixels = target.load()

    # Draw a 20x20 square at (50, 50)
    for i in range(50, 70):
        for j in range(50, 70):
            pixels[i, j] = 0

    # Draw a 30x30 square at (120, 50) (1.5x scale)
    for i in range(120, 150):
        for j in range(50, 80):
            pixels[i, j] = 0

    # Template is the 20x20 square with edge included
    template = target.crop((45, 45, 75, 75))

    # Search at multiple scales
    matches = matcher.find_matches_in_image(
        template, target, confidence_threshold=0.8, method="template", sizes=[1.0, 1.5], step=2
    )

    # Should find matches at both scales
    assert len(matches) >= 2

    # Check for different sized matches
    match_sizes = [(m.bbox[2] - m.bbox[0], m.bbox[3] - m.bbox[1]) for m in matches]

    # Should have 30x30 match (template with edges)
    assert any(abs(w - 30) <= 1 and abs(h - 30) <= 1 for w, h in match_sizes)

    # Should have ~45x45 match (30x30 template scaled by 1.5)
    assert any(abs(w - 45) <= 2 and abs(h - 45) <= 2 for w, h in match_sizes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
