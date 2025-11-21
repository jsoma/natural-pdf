from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from natural_pdf import PDF

DEFAULT_PDF = Path("pdfs/01-practice.pdf")


def measure(pdf_path: Path, page_index: int, iterations: int) -> list[float]:
    pdf = PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        manager = page._element_mgr  # noqa: SLF001 - benchmark uses internals intentionally
        durations: list[float] = []
        for _ in range(iterations):
            manager.invalidate_cache()
            start = time.perf_counter()
            manager.load_elements()
            durations.append(time.perf_counter() - start)
        return durations
    finally:
        pdf.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ElementManager.load_elements().")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to PDF file (default: {DEFAULT_PDF})",
    )
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index to benchmark.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of repeated load/invalidate cycles (default: 5).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        parser.error(f"PDF '{args.pdf}' does not exist.")

    durations = measure(args.pdf, args.page, args.iterations)
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    best = min(durations)
    worst = max(durations)

    print(f"PDF: {args.pdf} (page {args.page})")
    print(f"Iterations: {args.iterations}")
    print(f"Mean:   {mean:.4f}s")
    print(f"Median: {median:.4f}s")
    print(f"Best:   {best:.4f}s")
    print(f"Worst:  {worst:.4f}s")


if __name__ == "__main__":
    main()
