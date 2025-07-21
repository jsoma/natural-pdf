#!/usr/bin/env python3
"""
Natural PDF Performance Analysis Micro-Suite

This script analyzes memory usage and performance characteristics of Natural PDF
operations using real large PDFs to inform memory management decisions.
"""

import gc
import json
import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import psutil

import natural_pdf as npdf


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    python_objects: int
    operation: str
    page_count: int
    pdf_name: str
    additional_info: Dict[str, Any]


class PerformanceProfiler:
    """Profiles memory usage and performance of Natural PDF operations"""

    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        self.start_time = time.time()

        # Start tracemalloc for detailed Python memory tracking
        tracemalloc.start()

    def take_snapshot(
        self, operation: str, page_count: int = 0, pdf_name: str = "", **additional_info
    ):
        """Take a memory usage snapshot"""
        gc.collect()  # Force garbage collection for accurate measurement

        memory_info = self.process.memory_info()
        python_objects = len(gc.get_objects())

        snapshot = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            python_objects=python_objects,
            operation=operation,
            page_count=page_count,
            pdf_name=pdf_name,
            additional_info=additional_info,
        )

        self.snapshots.append(snapshot)
        print(
            f"[{snapshot.timestamp:.1f}s] {operation}: {snapshot.rss_mb:.1f}MB RSS, {python_objects} objects"
        )

    def save_results(self, test_name: str):
        """Save results to JSON and CSV"""
        # Convert to list of dicts for JSON serialization
        data = [asdict(s) for s in self.snapshots]

        # Save JSON
        json_path = self.output_dir / f"{test_name}_snapshots.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save CSV for easy analysis
        df = pd.DataFrame(data)
        csv_path = self.output_dir / f"{test_name}_snapshots.csv"
        df.to_csv(csv_path, index=False)

        print(f"Results saved to {json_path} and {csv_path}")
        return df


class PDFPerformanceTester:
    """Tests specific PDF operations and measures their performance"""

    def __init__(self, pdf_path: str, profiler: PerformanceProfiler):
        self.pdf_path = Path(pdf_path)
        self.pdf_name = self.pdf_path.stem
        self.profiler = profiler
        self.pdf = None

    def test_load_pdf(self):
        """Test just loading the PDF"""
        self.profiler.take_snapshot("before_load", pdf_name=self.pdf_name)

        self.pdf = npdf.PDF(str(self.pdf_path))

        self.profiler.take_snapshot(
            "after_load", pdf_name=self.pdf_name, total_pages=len(self.pdf.pages)
        )

    def test_page_access(self, max_pages: int = 10):
        """Test accessing pages sequentially"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            # Just access the page to trigger lazy loading
            _ = page.width, page.height

            self.profiler.take_snapshot(
                f"page_access_{i+1}",
                page_count=i + 1,
                pdf_name=self.pdf_name,
                page_width=page.width,
                page_height=page.height,
            )

    def test_describe_pages(self, max_pages: int = 5):
        """Test using .describe() on pages"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            # Use describe to understand page content
            try:
                description = page.describe()

                self.profiler.take_snapshot(
                    f"describe_{i+1}",
                    page_count=i + 1,
                    pdf_name=self.pdf_name,
                    description_length=len(description) if description else 0,
                )
            except Exception as e:
                self.profiler.take_snapshot(
                    f"describe_{i+1}_error", page_count=i + 1, pdf_name=self.pdf_name, error=str(e)
                )

    def test_element_collections(self, max_pages: int = 5):
        """Test find_all operations that create element collections"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            # Test different element collection operations
            operations = [
                ("words", lambda p: p.find_all("words")),
                ("text_elements", lambda p: p.find_all("text")),
                ("rects", lambda p: p.find_all("rect")),
                ("large_text", lambda p: p.find_all("text[size>12]")),
            ]

            for op_name, operation in operations:
                try:
                    elements = operation(page)
                    element_count = len(elements) if elements else 0

                    self.profiler.take_snapshot(
                        f"{op_name}_{i+1}",
                        page_count=i + 1,
                        pdf_name=self.pdf_name,
                        operation_type=op_name,
                        element_count=element_count,
                    )
                except Exception as e:
                    self.profiler.take_snapshot(
                        f"{op_name}_{i+1}_error",
                        page_count=i + 1,
                        pdf_name=self.pdf_name,
                        operation_type=op_name,
                        error=str(e),
                    )

    def test_image_generation(self, max_pages: int = 3, resolutions: List[int] = [72, 144, 216]):
        """Test image generation at different resolutions"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            for resolution in resolutions:
                try:
                    img = page.render(resolution=resolution)

                    self.profiler.take_snapshot(
                        f"image_{resolution}dpi_{i+1}",
                        page_count=i + 1,
                        pdf_name=self.pdf_name,
                        resolution=resolution,
                        image_size=f"{img.width}x{img.height}" if img else "None",
                    )

                    # Clean up image immediately to test memory release
                    del img

                except Exception as e:
                    self.profiler.take_snapshot(
                        f"image_{resolution}dpi_{i+1}_error",
                        page_count=i + 1,
                        pdf_name=self.pdf_name,
                        resolution=resolution,
                        error=str(e),
                    )

    def test_ocr(self, max_pages: int = 2):
        """Test OCR operations (expensive!)"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            try:
                # Run OCR
                page.apply_ocr(engine="easyocr")  # Default engine

                self.profiler.take_snapshot(
                    f"ocr_{i+1}", page_count=i + 1, pdf_name=self.pdf_name, operation_type="ocr"
                )

            except Exception as e:
                self.profiler.take_snapshot(
                    f"ocr_{i+1}_error",
                    page_count=i + 1,
                    pdf_name=self.pdf_name,
                    operation_type="ocr",
                    error=str(e),
                )

    def test_layout_analysis(self, max_pages: int = 3):
        """Test layout analysis operations"""
        if not self.pdf:
            self.test_load_pdf()

        pages_to_test = min(max_pages, len(self.pdf.pages))

        for i in range(pages_to_test):
            page = self.pdf.pages[i]

            try:
                # Run layout analysis
                layout_result = page.analyze_layout()

                self.profiler.take_snapshot(
                    f"layout_{i+1}",
                    page_count=i + 1,
                    pdf_name=self.pdf_name,
                    operation_type="layout",
                    layout_regions=len(layout_result) if layout_result else 0,
                )

            except Exception as e:
                self.profiler.take_snapshot(
                    f"layout_{i+1}_error",
                    page_count=i + 1,
                    pdf_name=self.pdf_name,
                    operation_type="layout",
                    error=str(e),
                )


def run_comprehensive_test(pdf_path: str, test_name: str):
    """Run a comprehensive test suite on a PDF"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST: {test_name}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}")

    profiler = PerformanceProfiler()
    tester = PDFPerformanceTester(pdf_path, profiler)

    # Initial baseline
    profiler.take_snapshot("baseline_start", pdf_name=Path(pdf_path).stem)

    # Test sequence
    print("\n1. Testing PDF Load...")
    tester.test_load_pdf()

    print("\n2. Testing Page Access...")
    tester.test_page_access(max_pages=10)

    print("\n3. Testing Describe Operations...")
    tester.test_describe_pages(max_pages=5)

    print("\n4. Testing Element Collections...")
    tester.test_element_collections(max_pages=5)

    print("\n5. Testing Image Generation...")
    tester.test_image_generation(max_pages=3)

    print("\n6. Testing Layout Analysis...")
    tester.test_layout_analysis(max_pages=3)

    # OCR test (only for image-heavy PDFs)
    if "OCR" in pdf_path or "image" in test_name.lower():
        print("\n7. Testing OCR (Image-heavy PDF)...")
        tester.test_ocr(max_pages=2)

    # Final snapshot
    profiler.take_snapshot("test_complete", pdf_name=Path(pdf_path).stem)

    # Save results
    df = profiler.save_results(test_name)

    # Quick analysis
    print(f"\n{'-'*40}")
    print("QUICK ANALYSIS:")
    print(f"Peak Memory: {df['rss_mb'].max():.1f} MB")
    print(f"Memory Growth: {df['rss_mb'].iloc[-1] - df['rss_mb'].iloc[0]:.1f} MB")
    print(f"Peak Objects: {df['python_objects'].max():,}")
    print(f"Total Time: {df['timestamp'].iloc[-1]:.1f} seconds")

    return df


def main():
    """Main test runner"""
    print("Natural PDF Performance Analysis Micro-Suite")
    print("=" * 50)

    # Find test PDFs
    large_pdfs_dir = Path("pdfs/hidden/large")
    if not large_pdfs_dir.exists():
        print(f"Error: {large_pdfs_dir} not found")
        print("Please ensure large test PDFs are available")
        return

    # Expected test PDFs
    test_pdfs = {
        "text_heavy": large_pdfs_dir / "appendix_fy2026.pdf",
        "image_heavy": large_pdfs_dir
        / "OCR 0802030-56.2022.8.14.0060_Cópia integral_Fazenda Marrocos.pdf",
    }

    results = {}

    for test_name, pdf_path in test_pdfs.items():
        if pdf_path.exists():
            try:
                results[test_name] = run_comprehensive_test(str(pdf_path), test_name)
            except Exception as e:
                print(f"Error testing {test_name}: {e}")
                traceback.print_exc()
        else:
            print(f"Warning: {pdf_path} not found, skipping {test_name} test")

    # Generate comparison report
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        for test_name, df in results.items():
            print(f"\n{test_name.upper()}:")
            print(f"  Peak Memory: {df['rss_mb'].max():.1f} MB")
            print(f"  Memory Growth: {df['rss_mb'].iloc[-1] - df['rss_mb'].iloc[0]:.1f} MB")
            print(f"  Peak Objects: {df['python_objects'].max():,}")
            print(f"  Duration: {df['timestamp'].iloc[-1]:.1f}s")

        print(f"\nResults saved to performance_results/ directory")
        print("Use the CSV files for detailed analysis")


if __name__ == "__main__":
    main()
