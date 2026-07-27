import os
import sys

import nox

# Documentation pages are executed by scripts/docs_build.py (see
# .github/workflows/docs.yml); there is no nox session for docs.

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
    "numpy<2.0",  # Must be last — PaddlePaddle incompatible with numpy 2.x
]


# Pinned so local runs and CI agree on formatting; bump deliberately and
# reformat the repo in the same commit.
LINT_TOOLS = ["black==26.5.1", "isort==7.0.0"]


@nox.session
def lint(session):
    """Run linters."""
    session.install(*LINT_TOOLS)
    session.run("black", "--check", ".")
    session.run("isort", "--check-only", ".")
    # Consider adding mypy checks if types are consistently added
    # session.run("mypy", "src", "tests") # Adjust paths as needed


@nox.session
def test_bare_install(session):
    """Smoke-test a bare `pip install natural-pdf` (no extras).

    Catches import-time reliance on optional dependencies and dependency
    creep in the core install. Unlike test_minimal, this installs NO test
    extras — only the package's own required dependencies.
    """
    session.install(".")
    session.run(
        "python",
        "-c",
        (
            "import natural_pdf; "
            "pdf = natural_pdf.PDF('pdfs/01-practice.pdf'); "
            "page = pdf.pages[0]; "
            "text = page.extract_text(); "
            "assert text and len(text) > 50, 'extract_text returned too little text'; "
            "el = page.find('text:contains(\"Durham\")'); "
            "assert el is not None, 'selector lookup failed'; "
            "tables = page.extract_table(); "
            "pdf.close(); "
            "print('bare-install smoke test OK')"
        ),
    )


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


# Optional: Add a test dependency group to pyproject.toml if needed
# [project.optional-dependencies]
# test = [
#     "pytest",
#     "pytest-cov", # Optional for coverage
# ]
