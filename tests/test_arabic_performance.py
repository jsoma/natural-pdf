#!/usr/bin/env python3
"""Test and profile performance differences between Arabic RTL and regular PDFs."""

import cProfile
import pstats
import time
from io import StringIO

import pytest

from natural_pdf import PDF


class TestArabicPerformance:
    """Test performance of Arabic/RTL PDF processing."""

    def profile_pdf_extraction(self, pdf_path: str, label: str):
        """Profile PDF text extraction and return timing info."""
        print(f"\n{'='*60}")
        print(f"Profiling: {label} ({pdf_path})")
        print("=" * 60)

        # Time the overall operation
        start_time = time.time()

        # Create profiler
        profiler = cProfile.Profile()

        # Profile the operations
        profiler.enable()

        pdf = PDF(pdf_path)
        page = pdf.pages[0]
        text = page.extract_text()

        profiler.disable()

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\nTotal time: {elapsed:.2f} seconds")
        print(f"Text length: {len(text)} characters")
        print(f"First 100 chars: {repr(text[:100])}")

        # Print profiling stats
        print("\nTop 20 time-consuming functions:")
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())

        # Also show stats by total time
        print("\nTop 20 by total time:")
        s2 = StringIO()
        ps2 = pstats.Stats(profiler, stream=s2).sort_stats("tottime")
        ps2.print_stats(20)
        print(s2.getvalue())

        return elapsed, len(text), text

    def test_arabic_vs_regular_performance(self):
        """Compare performance between Arabic and regular PDFs."""
        # Profile both PDFs
        arabic_time, arabic_len, arabic_text = self.profile_pdf_extraction(
            "pdfs/arabic.pdf", "Arabic RTL PDF"
        )
        regular_time, regular_len, regular_text = self.profile_pdf_extraction(
            "pdfs/types-of-type.pdf", "Regular PDF"
        )

        # Summary comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print(
            f"Arabic PDF:  {arabic_time:.2f}s for {arabic_len} chars ({arabic_len/arabic_time:.0f} chars/sec)"
        )
        print(
            f"Regular PDF: {regular_time:.2f}s for {regular_len} chars ({regular_len/regular_time:.0f} chars/sec)"
        )
        slowdown = arabic_time / regular_time
        print(f"Slowdown factor: {slowdown:.1f}x slower")

        # This test can be used to track performance regressions
        # For now, we'll just assert that both PDFs can be processed
        assert arabic_len > 0, "Arabic PDF should contain text"
        assert regular_len > 0, "Regular PDF should contain text"

        # Log a warning if Arabic is significantly slower
        if slowdown > 5:
            print(f"\nWARNING: Arabic PDF processing is {slowdown:.1f}x slower than expected!")


if __name__ == "__main__":
    # Run with detailed output
    pytest.main([__file__, "-v", "-s"])
