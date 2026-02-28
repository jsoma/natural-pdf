import os
import sys

import nox

# ============================================================================
# DOCUMENTATION WORKFLOW
# ============================================================================
#
# Common commands for documentation and tutorials:
#
# 1. Execute notebooks + run tests (recommended):
#    nox -s docs
#
# 2. Force re-execute all notebooks + run tests:
#    nox -s docs-force
#
# 3. Advanced: Execute notebooks only with custom options:
#    python scripts/01-execute_notebooks.py --force --workers 8
#
# ============================================================================

# Ensure nox uses the same Python version you are developing with or whichever is appropriate
# Make sure this Python version has nox installed (`pip install nox`)
# You can specify multiple Python versions to test against, e.g., ["3.10", "3.11", "3.12"]
nox.options.sessions = ["lint", "test_minimal", "test_full"]
nox.options.reuse_existing_virtualenvs = True  # Faster runs by reusing environments
nox.options.default_venv_backend = "uv"  # Use uv for faster venv creation and package installation

PYTHON_VERSIONS = (
    ["3.10", "3.11", "3.12"] if sys.platform != "darwin" else ["3.10", "3.11", "3.12"]
)  # Add more as needed

# Packages that are not part of the core install but are needed for full functionality
# This list is used for the 'test_full' session
OPTIONAL_PACKAGES = [
    "ipywidgets>=7.0.0,<10.0.0",
    "easyocr",
    "paddleocr",
    "paddlepaddle",
    "paddlex[ocr]>=3.0.2",
    "numpy<2.0",  # PaddlePaddle incompatible with numpy 2.x scalar conversion
    "surya-ocr<0.15",
    "doclayout_yolo",
    # NOTE: python-doctr requires huggingface_hub<1.0 due to deprecated Repository import
    # This pins transformers to 4.x (5.0+ requires hf-hub>=1.3). Remove when doctr fixes upstream:
    # https://github.com/mindee/doctr/pull/2024
    "huggingface_hub>=0.23.0,<1.0",
    "python-doctr[torch]",
    "openai",
    "lancedb",
    "pyarrow",
    "img2pdf",
    "jupytext",
    "nbformat",
]


@nox.session
def lint(session):
    """Run linters."""
    session.install("black", "isort")
    session.run("black", "--check", ".")
    session.run("isort", "--check-only", ".")
    # Consider adding mypy checks if types are consistently added
    # session.run("mypy", "src", "tests") # Adjust paths as needed


@nox.session
def test_minimal(session):
    """Run tests with only core dependencies, expecting failures for optional features."""
    session.install(".[test]")
    # Skip tutorial, QA, and optional dependency suites to keep this environment lightweight
    session.run(
        "pytest",
        "tests",
        "-n",
        "auto",
        "-m",
        "not tutorial and not qa and not optional_deps",
    )


@nox.session
def test_full(session):
    """Run tests with all optional dependencies installed."""
    # Install the main package with test dependencies first
    session.install(".[test]")

    # On Windows in CI, pre-install torch from official PyTorch wheel to avoid DLL issues
    if sys.platform.startswith("win") and "GITHUB_ACTIONS" in os.environ:
        session.log("Pre-installing torch from official PyTorch wheel to avoid shm.dll error")
        session.install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    # Install all optional packages
    # Using separate install commands can help with complex dependencies
    for package in OPTIONAL_PACKAGES:
        # Special handling for paddle on macOS if necessary, though often it works now
        # if "paddle" in package and session.platform == "darwin":
        #     session.log(f"Skipping {package} on macOS for now.")
        #     continue
        session.install(package)

    # Run tests with all dependencies available
    session.run("pytest", "tests", "-n", "auto", "-m", "not tutorial")


@nox.session(name="docs", python="3.10")
def docs(session):
    """Execute markdown tutorials and run tutorial tests in one command.

    This replaces the old two-step process:
    - OLD: python scripts/01-execute_notebooks.py && nox -s tutorials
    - NEW: nox -s docs

    Uses intelligent caching to skip unchanged notebooks.
    """
    # Install all dependencies needed for both notebook execution and testing
    session.install(".[all,dev]")

    # On Windows in CI, pre-install torch from official PyTorch wheel to avoid DLL issues
    if sys.platform.startswith("win") and "GITHUB_ACTIONS" in os.environ:
        session.log("Pre-installing torch from official PyTorch wheel to avoid shm.dll error")
        session.install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    session.install("surya-ocr<0.15")
    session.install("easyocr")
    session.install("doclayout_yolo")
    for package in OPTIONAL_PACKAGES:
        session.install(package)

    # First, execute notebooks (convert md to ipynb and run them)
    session.log("Step 1: Executing markdown notebooks...")
    workers = os.environ.get("NOTEBOOK_WORKERS", str(os.cpu_count() or 4))
    session.run("python", "scripts/01-execute_notebooks.py", "--workers", workers)

    # Then run tutorial tests
    # Note: These tests verify the notebooks were executed successfully,
    # they do NOT re-execute the notebooks (that would be redundant)
    session.log("Step 2: Running tutorial tests...")
    session.run("pytest", "tests", "-m", "tutorial", "-n", workers, "-v", "--tb=short")


@nox.session(name="test-docs", python="3.10")
def test_docs(session):
    """Test markdown code examples in documentation.

    This runs mktestdocs to verify that code examples in docs/getting-started/
    and docs/cookbook/ are actually executable. Catches documentation rot early.

    Usage:
        nox -s test-docs
    """
    session.install(".[test]")
    session.run("pytest", "tests/test_markdown_docs.py", "-v")


@nox.session(name="docs-force", python="3.10")
def docs_force(session):
    """Force execute all markdown tutorials and run tutorial tests.

    Use this when you want to re-execute ALL notebooks regardless of cache:
    - nox -s docs-force

    This is useful when dependencies change or for clean rebuilds.
    """
    # Install all dependencies
    session.install(".[all,dev]")

    # On Windows in CI, pre-install torch from official PyTorch wheel to avoid DLL issues
    if sys.platform.startswith("win") and "GITHUB_ACTIONS" in os.environ:
        session.log("Pre-installing torch from official PyTorch wheel to avoid shm.dll error")
        session.install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    session.install("surya-ocr<0.15")
    session.install("easyocr")
    session.install("doclayout_yolo")
    for package in OPTIONAL_PACKAGES:
        session.install(package)

    # Execute notebooks with --force flag
    session.log("Step 1: Force executing all markdown notebooks...")
    workers = os.environ.get("NOTEBOOK_WORKERS", str(os.cpu_count() or 4))
    session.run("python", "scripts/01-execute_notebooks.py", "--force", "--workers", workers)

    # Run tutorial tests
    # Note: These tests verify the notebooks were executed successfully,
    # they do NOT re-execute the notebooks (that would be redundant)
    session.log("Step 2: Running tutorial tests...")
    session.run("pytest", "tests", "-m", "tutorial", "-n", workers, "-v", "--tb=short")


# Optional: Add a test dependency group to pyproject.toml if needed
# [project.optional-dependencies]
# test = [
#     "pytest",
#     "pytest-cov", # Optional for coverage
# ]
