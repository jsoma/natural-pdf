"""
Test markdown documentation code examples using mktestdocs.

This module validates that code examples in the documentation actually work.
Tests focus on getting-started docs which use the practice PDF and should be fully runnable.

Usage:
    uv run pytest tests/test_markdown_docs.py -v
    uv run pytest tests/test_markdown_docs.py -k "quickstart"
"""

import pathlib

import pytest

# mktestdocs is an optional dependency for doc testing
mktestdocs = pytest.importorskip("mktestdocs")


DOCS_DIR = pathlib.Path("docs")

# Only test docs that use pdfs/01-practice.pdf and are fully runnable
# Cookbook docs are illustrative and reference non-existent files
RUNNABLE_DOCS = [
    DOCS_DIR / "getting-started" / "quickstart.md",
    DOCS_DIR / "getting-started" / "concepts.md",
]

# Filter to only existing files
TESTABLE_DOCS = [f for f in RUNNABLE_DOCS if f.exists()]


@pytest.mark.parametrize("fpath", TESTABLE_DOCS, ids=lambda x: x.name)
def test_markdown_examples(fpath):
    """Test that code examples in markdown files are executable.

    Uses memory=True to preserve state between code blocks, allowing
    tutorials to build on previous examples.
    """
    mktestdocs.check_md_file(fpath=fpath, memory=True)


# Smoke test to ensure the test infrastructure works
def test_mktestdocs_importable():
    """Verify mktestdocs is properly installed."""
    assert hasattr(mktestdocs, "check_md_file")
