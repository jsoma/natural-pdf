"""Test perceptual hash with white masking"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from natural_pdf.vision.similarity import VisualMatcher, compute_phash, hash_similarity


def test_phash_masking_basic():
    """Test that phash masking improves matching across different backgrounds"""

    # Create logo on white background
    img1 = Image.new("L", (64, 64), color=255)  # White
    draw = ImageDraw.Draw(img1)
    # Draw a simple cross pattern
    draw.rectangle([20, 10, 44, 20], fill=0)  # Horizontal bar
    draw.rectangle([27, 5, 37, 54], fill=0)  # Vertical bar

    # Same logo on gray background
    img2 = Image.new("L", (64, 64), color=128)  # Gray
    draw = ImageDraw.Draw(img2)
    draw.rectangle([20, 10, 44, 20], fill=0)  # Same cross
    draw.rectangle([27, 5, 37, 54], fill=0)

    # Same logo on dark background
    img3 = Image.new("L", (64, 64), color=64)  # Dark gray
    draw = ImageDraw.Draw(img3)
    draw.rectangle([20, 10, 44, 20], fill=0)  # Same cross
    draw.rectangle([27, 5, 37, 54], fill=0)

    # Compute hashes without masking
    hash1_no_mask = compute_phash(img1, hash_size=8)
    hash2_no_mask = compute_phash(img2, hash_size=8)
    hash3_no_mask = compute_phash(img3, hash_size=8)

    # Similarity without masking
    sim_12_no_mask = hash_similarity(hash1_no_mask, hash2_no_mask)
    sim_13_no_mask = hash_similarity(hash1_no_mask, hash3_no_mask)

    # Compute hashes WITH masking (mask white/near-white pixels)
    hash1_masked = compute_phash(img1, hash_size=8, mask_threshold=240)  # 240/255 ≈ 0.94
    hash2_masked = compute_phash(img2, hash_size=8, mask_threshold=240)
    hash3_masked = compute_phash(img3, hash_size=8, mask_threshold=240)

    # Similarity with masking
    sim_12_masked = hash_similarity(hash1_masked, hash2_masked)
    sim_13_masked = hash_similarity(hash1_masked, hash3_masked)

    # With masking, similarity should be much higher (or already perfect)
    assert (
        sim_12_masked >= sim_12_no_mask
    ), f"Masked similarity {sim_12_masked:.3f} should be at least as good as {sim_12_no_mask:.3f}"
    assert (
        sim_13_masked >= sim_13_no_mask
    ), f"Masked similarity {sim_13_masked:.3f} should be at least as good as {sim_13_no_mask:.3f}"

    # All masked hashes should be very similar (same logo, different backgrounds)
    assert sim_12_masked > 0.95  # Should be nearly identical
    assert sim_13_masked > 0.95


def test_phash_masking_text():
    """Test phash masking with text-like patterns"""

    # Create "text" on white background
    img_white = Image.new("L", (100, 40), color=255)
    draw = ImageDraw.Draw(img_white)
    # Draw some "text" (rectangles simulating characters)
    for i, x in enumerate([10, 25, 40, 55, 70, 85]):
        draw.rectangle([x, 10, x + 10, 30], fill=0)

    # Same "text" on colored background
    img_colored = Image.new("L", (100, 40), color=200)  # Light gray
    draw = ImageDraw.Draw(img_colored)
    for i, x in enumerate([10, 25, 40, 55, 70, 85]):
        draw.rectangle([x, 10, x + 10, 30], fill=0)

    # Without masking
    hash_white = compute_phash(img_white)
    hash_colored = compute_phash(img_colored)
    sim_no_mask = hash_similarity(hash_white, hash_colored)

    # With masking (ignore light pixels)
    hash_white_masked = compute_phash(img_white, mask_threshold=180)
    hash_colored_masked = compute_phash(img_colored, mask_threshold=180)
    sim_masked = hash_similarity(hash_white_masked, hash_colored_masked)

    # Masking should improve similarity (or maintain if already perfect)
    assert sim_masked >= sim_no_mask
    # If not already perfect, masking should help
    if sim_no_mask < 0.99:
        assert sim_masked > sim_no_mask
    assert sim_masked > 0.90  # Should be very similar


def test_visual_matcher_phash_masking():
    """Test VisualMatcher with phash and masking"""
    matcher = VisualMatcher()

    # Template: logo on white
    template = Image.new("L", (50, 50), color=255)
    draw = ImageDraw.Draw(template)
    # Draw a circle
    draw.ellipse([10, 10, 40, 40], fill=0)
    # Draw inner circle
    draw.ellipse([20, 20, 30, 30], fill=255)

    # Target: has same logo on different backgrounds
    target = Image.new("L", (200, 100), color=128)  # Gray background

    # Place logo at different positions with different backgrounds
    # Position 1: on gray (default background)
    draw = ImageDraw.Draw(target)
    draw.ellipse([25, 25, 55, 55], fill=0)
    draw.ellipse([35, 35, 45, 45], fill=128)

    # Position 2: on light background
    draw.rectangle([100, 20, 160, 80], fill=220)  # Light background
    draw.ellipse([115, 25, 145, 55], fill=0)
    draw.ellipse([125, 35, 135, 45], fill=220)

    # Find matches WITHOUT masking
    matches_no_mask = matcher.find_matches_in_image(
        template, target, confidence_threshold=0.7, method="phash"
    )

    # Find matches WITH masking
    matches_masked = matcher.find_matches_in_image(
        template,
        target,
        confidence_threshold=0.7,
        method="phash",
        mask_threshold=0.9,  # This gets converted to 0.9*255 internally
    )

    # Both should find matches
    assert len(matches_no_mask) >= 1
    assert len(matches_masked) >= 1

    # Masked matching should find at least as many matches
    assert len(matches_masked) >= len(matches_no_mask)

    # For logos with different backgrounds, masking helps find more consistent matches
    print(f"Without masking: found {len(matches_no_mask)} matches")
    print(f"With masking: found {len(matches_masked)} matches")


def test_edge_cases():
    """Test edge cases for phash masking"""

    # All white image
    img_white = Image.new("L", (50, 50), color=255)
    hash_white = compute_phash(img_white, mask_threshold=250)
    # Should still produce a hash (uses median of non-masked pixels)
    assert isinstance(hash_white, int)

    # Gradient image
    img_gradient = Image.new("L", (50, 50))
    pixels = img_gradient.load()
    for i in range(50):
        for j in range(50):
            pixels[i, j] = int(i * 255 / 49)

    # Different thresholds should give different hashes
    hash_150 = compute_phash(img_gradient, mask_threshold=150)
    hash_200 = compute_phash(img_gradient, mask_threshold=200)
    hash_250 = compute_phash(img_gradient, mask_threshold=250)

    # Should be different as they mask different amounts
    assert hash_150 != hash_200 or hash_200 != hash_250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
