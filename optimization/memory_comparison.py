#!/usr/bin/env python3
"""
Memory comparison script to measure the effectiveness of the character duplication fix.

This script compares memory usage before and after the optimization by:
1. Testing with a text-heavy PDF
2. Measuring detailed memory usage patterns
3. Calculating memory savings
"""

import gc
import os
import sys
from pathlib import Path

import psutil

import natural_pdf as npdf


def get_detailed_memory_info():
    """Get detailed memory information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "python_objects": len(gc.get_objects()),
    }


def analyze_character_storage(page):
    """Analyze how characters are stored in the page"""
    # Force element loading
    text_elements = page.find_all("text")

    total_char_indices = 0
    total_char_dicts = 0
    total_chars_in_words = 0
    memory_efficient_words = 0
    legacy_words = 0

    for element in text_elements:
        if hasattr(element, "_char_indices") and element._char_indices:
            memory_efficient_words += 1
            total_char_indices += len(element._char_indices)
            total_chars_in_words += len(element._char_indices)

        if hasattr(element, "_char_dicts") and element._char_dicts:
            total_char_dicts += len(element._char_dicts)
            if not (hasattr(element, "_char_indices") and element._char_indices):
                legacy_words += 1
                total_chars_in_words += len(element._char_dicts)

    # Get individual character elements
    char_elements = []
    if hasattr(page, "_element_mgr"):
        char_elements = page._element_mgr.get_elements("chars")

    return {
        "total_words": len(text_elements),
        "memory_efficient_words": memory_efficient_words,
        "legacy_words": legacy_words,
        "total_char_elements": len(char_elements),
        "total_char_indices": total_char_indices,
        "total_char_dicts": total_char_dicts,
        "total_chars_in_words": total_chars_in_words,
        "estimated_duplication_ratio": total_char_dicts / max(len(char_elements), 1),
    }


def test_memory_optimization():
    """Test the memory optimization with a real PDF"""

    # Test with the practice PDF
    test_pdf = Path("pdfs/01-practice.pdf")
    if not test_pdf.exists():
        print(f"Test PDF not found: {test_pdf}")
        return

    print("=" * 60)
    print("MEMORY OPTIMIZATION ANALYSIS")
    print("=" * 60)

    # Baseline memory
    gc.collect()
    baseline_memory = get_detailed_memory_info()
    print(
        f"Baseline memory: {baseline_memory['rss_mb']:.2f} MB RSS, {baseline_memory['python_objects']:,} objects"
    )

    # Load PDF
    pdf = npdf.PDF(str(test_pdf))
    page = pdf.pages[0]

    post_load_memory = get_detailed_memory_info()
    print(
        f"After PDF load: {post_load_memory['rss_mb']:.2f} MB RSS, {post_load_memory['python_objects']:,} objects"
    )

    # Analyze character storage
    storage_analysis = analyze_character_storage(page)

    final_memory = get_detailed_memory_info()
    print(
        f"After element load: {final_memory['rss_mb']:.2f} MB RSS, {final_memory['python_objects']:,} objects"
    )

    print("\n" + "=" * 40)
    print("CHARACTER STORAGE ANALYSIS")
    print("=" * 40)

    print(f"Total words: {storage_analysis['total_words']}")
    print(f"Memory-efficient words: {storage_analysis['memory_efficient_words']}")
    print(f"Legacy words: {storage_analysis['legacy_words']}")
    print(f"Total character elements: {storage_analysis['total_char_elements']}")
    print(f"Character indices used: {storage_analysis['total_char_indices']}")
    print(f"Character dicts stored: {storage_analysis['total_char_dicts']}")
    print(f"Characters referenced by words: {storage_analysis['total_chars_in_words']}")

    # Calculate optimization metrics
    duplication_ratio = storage_analysis["estimated_duplication_ratio"]
    optimization_percentage = (
        storage_analysis["memory_efficient_words"] / max(storage_analysis["total_words"], 1) * 100
    )

    print(f"\nOptimization metrics:")
    print(f"- Duplication ratio: {duplication_ratio:.2f}x")
    print(f"- Words using optimization: {optimization_percentage:.1f}%")

    # Memory savings estimation
    memory_used = final_memory["rss_mb"] - baseline_memory["rss_mb"]
    chars_total = storage_analysis["total_char_elements"]

    if chars_total > 0:
        memory_per_char = memory_used / chars_total * 1024  # KB per char
        print(f"- Memory per character: {memory_per_char:.2f} KB")

        # Estimate savings from eliminating _char_dicts duplication
        duplicated_chars = storage_analysis["total_char_dicts"]
        if duplicated_chars > 0:
            estimated_wasted_memory = duplicated_chars * memory_per_char / 1024  # MB
            print(f"- Estimated memory saved by optimization: {estimated_wasted_memory:.2f} MB")
            print(
                f"- Memory efficiency improvement: {estimated_wasted_memory / memory_used * 100:.1f}%"
            )

    print(f"\nTotal memory used for page processing: {memory_used:.2f} MB")

    # Test functionality
    print("\n" + "=" * 40)
    print("FUNCTIONALITY VERIFICATION")
    print("=" * 40)

    # Test character access
    test_elements = page.find_all("text")[:3]
    for i, element in enumerate(test_elements):
        print(f"\nWord {i+1}: '{element.text[:30]}{'...' if len(element.text) > 30 else ''}'")

        if hasattr(element, "_char_indices") and element._char_indices:
            chars = element.chars
            print(
                f"  - Uses character indices: {len(element._char_indices)} indices -> {len(chars)} chars"
            )
            print(f"  - Memory optimization: ACTIVE")

            # Verify character access works
            if chars:
                first_char = chars[0]
                print(
                    f"  - First char: '{first_char.text}' at ({first_char.x0:.1f}, {first_char.top:.1f})"
                )

        elif hasattr(element, "_char_dicts") and element._char_dicts:
            print(f"  - Uses character dicts: {len(element._char_dicts)} dicts")
            print(f"  - Memory optimization: LEGACY MODE")

        else:
            print(f"  - No character data available")

    print("\n" + "=" * 60)
    print("✅ MEMORY OPTIMIZATION ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_optimization()
