"""Test template matching with white masking"""

import numpy as np
import pytest
from PIL import Image

from natural_pdf.vision.similarity import VisualMatcher
from natural_pdf.vision.template_matching import TemplateMatcher


def test_white_masking_basic():
    """Test that white masking ignores white pixels"""
    matcher = TemplateMatcher(method="zncc")

    # Create template: black square on white background
    template = np.ones((20, 20), dtype=np.float32)  # White
    template[5:15, 5:15] = 0.0  # Black square in center

    # Create target with same black square but on gray background
    target = np.ones((100, 100), dtype=np.float32) * 0.5  # Gray background
    target[40:60, 40:60] = 0.0  # Black square at (40, 40)

    # Without masking, the match should be poor due to different backgrounds
    scores_no_mask = matcher.match_template(target, template, step=1)
    max_score_no_mask = np.max(scores_no_mask)

    # With masking white pixels (>= 0.95), should find perfect match
    scores_masked = matcher.match_template(target, template, step=1, mask_threshold=0.95)
    max_score_masked = np.max(scores_masked)
    max_y, max_x = np.unravel_index(np.argmax(scores_masked), scores_masked.shape)

    # Masked matching should give much better score
    assert max_score_masked > max_score_no_mask + 0.2
    assert max_score_masked > 0.99  # Should be nearly perfect

    # Should find the correct location
    # Note: With masking, we're matching just the 10x10 black square
    # Template position (35,35) + black square offset (5,5) = target square at (40,40)
    assert 35 <= max_x <= 45 and 35 <= max_y <= 45  # Multiple positions give perfect match


def test_white_masking_different_backgrounds():
    """Test finding logo on different colored backgrounds"""
    matcher = TemplateMatcher(method="zncc")

    # Create a simple logo template (cross on white)
    template = np.ones((30, 30), dtype=np.float32)  # White
    # Vertical line
    template[5:25, 14:16] = 0.0
    # Horizontal line
    template[14:16, 5:25] = 0.0

    # Create target with same logo on different backgrounds
    target = np.zeros((100, 200), dtype=np.float32)
    # Logo on black background (left)
    target[20:50, 20:50] = template  # Direct copy
    # Logo on gray background (middle)
    target[20:50, 80:110] = 0.5  # Gray
    target[25:45, 94:96] = 0.0  # Vertical
    target[34:36, 85:105] = 0.0  # Horizontal
    # Logo on colored/noisy background (right)
    target[20:50, 140:170] = 0.7  # Light gray
    target[25:45, 154:156] = 0.0  # Vertical
    target[34:36, 145:165] = 0.0  # Horizontal

    # Search with white masking
    scores = matcher.match_template(target, template, step=2, mask_threshold=0.95)

    # Find peaks
    threshold = 0.8
    y_indices, x_indices = np.where(scores >= threshold)

    # Should find all three logos
    assert len(x_indices) >= 3

    # Check approximate locations (with step=2 tolerance)
    expected_x = [20, 80, 140]
    for exp_x in expected_x:
        found = any(abs(x * 2 - exp_x) <= 2 for x in x_indices)
        assert found, f"Logo at x={exp_x} not found"


def test_all_white_template():
    """Test edge case where template is all white"""
    matcher = TemplateMatcher()

    # All white template
    template = np.ones((10, 10), dtype=np.float32)
    target = np.random.rand(50, 50).astype(np.float32)

    # With mask threshold, all pixels are masked - should return zeros
    scores = matcher.match_template(target, template, mask_threshold=0.95)

    assert np.all(scores == 0)


def test_visual_matcher_with_masking():
    """Test VisualMatcher integration with white masking"""
    matcher = VisualMatcher()

    # Create images with logo
    # Template: black logo on white
    template_img = Image.new("L", (40, 40), color=255)
    pixels = template_img.load()
    # Draw a simple shape
    for i in range(10, 30):
        for j in range(10, 30):
            if abs(i - 20) + abs(j - 20) < 10:  # Diamond shape
                pixels[i, j] = 0

    # Target: same logo on gray background
    target_img = Image.new("L", (200, 200), color=128)
    pixels = target_img.load()
    # Place logo at two locations
    for i in range(10, 30):
        for j in range(10, 30):
            if abs(i - 20) + abs(j - 20) < 10:
                pixels[50 + i, 50 + j] = 0
                pixels[100 + i, 120 + j] = 0

    # Find matches with white masking
    matches = matcher.find_matches_in_image(
        template_img,
        target_img,
        confidence_threshold=0.8,
        method="template",
        mask_threshold=0.95,  # Mask white pixels
        step=2,
    )

    # Should find both logos
    assert len(matches) >= 2

    # Check locations
    match_locs = [(m.bbox[0], m.bbox[1]) for m in matches]

    # First logo around (50, 50)
    assert any(abs(x - 50) <= 2 and abs(y - 50) <= 2 for x, y in match_locs)

    # Second logo around (100, 120)
    assert any(abs(x - 100) <= 2 and abs(y - 120) <= 2 for x, y in match_locs)


def test_mask_threshold_values():
    """Test different mask threshold values"""
    matcher = TemplateMatcher()

    # Create template with meaningful pattern and gradient background
    template = np.ones((20, 20), dtype=np.float32)
    # Add gradient background
    for i in range(20):
        template[:, i] = 0.7 + (i / 19.0) * 0.3  # Gradient from 0.7 to 1.0
    # Add dark pattern
    template[8:12, 8:12] = 0.1  # Dark square in center

    # Target with similar pattern
    target = np.ones((50, 50), dtype=np.float32) * 0.5
    target[20:40, 20:40] = template  # Place template in target

    # Different thresholds should give different results
    scores_70 = matcher.match_template(target, template, mask_threshold=0.7)
    scores_85 = matcher.match_template(target, template, mask_threshold=0.85)
    scores_95 = matcher.match_template(target, template, mask_threshold=0.95)

    # Get max scores
    max_70 = np.max(scores_70)
    max_85 = np.max(scores_85)
    max_95 = np.max(scores_95)

    # With threshold 0.7, all pixels are used (no masking)
    # With threshold 0.95, only left part of gradient and dark square are used
    # Scores should be different
    assert max_70 != max_85, f"Scores should differ: {max_70} vs {max_85}"
    assert max_85 != max_95, f"Scores should differ: {max_85} vs {max_95}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
