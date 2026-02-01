"""
Natural PDF LLM Extraction Benchmark

Compare LLM vision-based PDF extraction against Natural PDF's deterministic extraction.
Detects hallucination, autocomplete errors, and normalization issues.

Usage:
    python -m benchmark prepare PDF_NAME              # Generate page images + ground truth
    python -m benchmark evaluate PDF_NAME --models    # Run LLM evaluations
    python -m benchmark report PDF_NAME               # Generate HTML report
    python -m benchmark run [--models MODEL1,MODEL2]  # Run full pipeline
    python -m benchmark list                          # List available PDF configs
"""

from benchmark.config import BenchmarkConfig
from benchmark.providers import get_provider
from benchmark.runner import BenchmarkRunner
from benchmark.schemas import GroundTruth, LLMResult, PageGroundTruth

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "GroundTruth",
    "LLMResult",
    "PageGroundTruth",
    "get_provider",
]
