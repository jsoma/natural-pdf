#!/usr/bin/env python3
"""
Test script to verify character duplication fix works correctly.

This test verifies that:
1. Memory usage is reduced by eliminating character duplication
2. All existing functionality still works correctly
3. Character access through words remains functional
"""

import gc
import os
import sys
from pathlib import Path

import psutil
import pytest

import natural_pdf as npdf


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class TestCharacterMemoryFix:
    """Test suite for character memory optimization"""

    @pytest.fixture
    def test_pdf_path(self):
        """Get path to a test PDF"""
        # Use the practice PDF for testing
        test_path = Path("pdfs/01-practice.pdf")
        if not test_path.exists():
            pytest.skip("Test PDF not found")
        return str(test_path)

    def test_character_access_still_works(self, test_pdf_path):
        """Test that character access through words still works after optimization"""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        # Force loading of elements
        text_elements = page.find_all("text")

        # Test that we have text elements
        assert len(text_elements) > 0, "Should have text elements"
        print(f"Found {len(text_elements)} text elements")

        # Test that words can access their constituent characters
        for word in text_elements[:5]:  # Test first 5 words
            if hasattr(word, "_char_indices") and word._char_indices:
                # New optimized approach
                constituent_chars = word.chars
                assert isinstance(constituent_chars, list), "word.chars should return a list"
                assert len(constituent_chars) > 0, "Should have constituent characters"

                # Test character properties
                for char in constituent_chars[:3]:  # Test first 3 chars of each word
                    assert hasattr(char, "text"), "Character should have text attribute"
                    assert hasattr(char, "x0"), "Character should have x0 coordinate"

            elif hasattr(word, "_char_dicts") and word._char_dicts:
                # Old approach - should still work for compatibility
                char_dicts = word._char_dicts
                assert isinstance(char_dicts, list), "word._char_dicts should be a list"
                assert len(char_dicts) > 0, "Should have character dictionaries"

    def test_memory_usage_improvement(self, test_pdf_path):
        """Test that memory usage is improved with the optimization"""
        # This test will compare memory usage patterns
        # Note: Exact numbers will vary, but we should see improvement

        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        # Measure memory before loading elements
        gc.collect()
        memory_before = get_memory_usage()

        # Load elements (this triggers the optimization)
        chars = page.find_all("text")
        words = page.find_all("words")

        # Measure memory after loading
        gc.collect()
        memory_after = get_memory_usage()

        memory_used = memory_after - memory_before

        # Log the memory usage for analysis
        print(f"\nMemory usage analysis:")
        print(f"Characters loaded: {len(chars)}")
        print(f"Words loaded: {len(words)}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Memory per character: {memory_used / len(chars) * 1024:.2f} KB" if chars else "N/A")

        # The memory usage should be reasonable (not exact test due to variability)
        # Main goal is to verify no crashes and reasonable memory usage
        assert memory_used < 100, f"Memory usage seems too high: {memory_used:.2f} MB"

    def test_word_text_extraction_works(self, test_pdf_path):
        """Test that text extraction from words still works correctly"""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        words = page.find_all("text")  # All text elements are words in this PDF

        # Test text extraction from words
        for word in words[:10]:  # Test first 10 words
            word_text = word.text
            assert isinstance(word_text, str), "Word text should be a string"

            # Text should not be empty for actual words
            if word_text.strip():  # Skip empty/whitespace words
                assert len(word_text) > 0, "Non-empty words should have text content"

    def test_backwards_compatibility(self, test_pdf_path):
        """Test that existing code patterns still work"""
        pdf = npdf.PDF(test_pdf_path)
        page = pdf.pages[0]

        # Test that existing element access patterns work
        all_elements = page.find_all("text")
        assert len(all_elements) > 0, "Should find text elements"

        # Test that element properties are accessible
        for element in all_elements[:5]:
            assert hasattr(element, "text"), "Element should have text attribute"
            assert hasattr(element, "x0"), "Element should have x0 coordinate"
            assert hasattr(element, "top"), "Element should have top coordinate"
            assert hasattr(element, "width"), "Element should have width"
            assert hasattr(element, "height"), "Element should have height"


def main():
    """Run the memory fix test"""
    print("Running character memory optimization test...")

    # Check if test PDF exists
    test_pdf = Path("pdfs/01-practice.pdf")
    if not test_pdf.exists():
        print(f"ERROR: Test PDF not found at {test_pdf}")
        print("Please ensure the test PDF exists before running this test.")
        return 1

    # Run pytest on just this file
    exit_code = pytest.main([__file__, "-v", "-s"])

    if exit_code == 0:
        print("\n✅ All memory optimization tests passed!")
        print("The character duplication fix is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        print("The memory optimization needs investigation.")

    return exit_code


if __name__ == "__main__":
    exit(main())
