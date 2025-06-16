import os
import sys

import nox

# Ensure nox uses the same Python version you are developing with or whichever is appropriate
# Make sure this Python version has nox installed (`pip install nox`)
# You can specify multiple Python versions to test against, e.g., ["3.9", "3.10", "3.11"]
nox.options.sessions = ["lint", "test_minimal", "test_full"]
nox.options.reuse_existing_virtualenvs = True  # Faster runs by reusing environments
nox.options.default_venv_backend = "uv"  # Use uv for faster venv creation and package installation

PYTHON_VERSIONS = (
    ["3.9", "3.10", "3.11"] if sys.platform != "darwin" else ["3.9", "3.10", "3.11"]
)  # Add more as needed

# Packages that are not part of the core install but are needed for full functionality
# This list is used for the 'test_full' session
OPTIONAL_PACKAGES = [
    "ipywidgets>=7.0.0,<10.0.0",
    "easyocr",
    "paddleocr",
    "paddlepaddle",
    "surya-ocr",
    "doclayout_yolo",
    "python-doctr[torch]",
    "docling",
    "openai",
    "lancedb",
    "pyarrow",
    "deskew>=1.5",
    "img2pdf",
    "jupytext",
    "nbformat",
]


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    """Run linters."""
    session.install("black", "isort")
    session.run("black", "--check", ".")
    session.run("isort", "--check-only", ".")
    # Consider adding mypy checks if types are consistently added
    # session.run("mypy", "src", "tests") # Adjust paths as needed


@nox.session(python=PYTHON_VERSIONS)
def test_minimal(session):
    """Run tests with only core dependencies, expecting failures for optional features."""
    session.install(".[test]")
    session.run("pytest", "tests", "-n", "auto", "-m", "not tutorial")


@nox.session(python=PYTHON_VERSIONS)
def test_full(session):
    """Run tests with all optional dependencies installed."""
    # Install the main package with test dependencies first
    session.install(".[test]")

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


@nox.session(python=PYTHON_VERSIONS)
def test_favorites(session):
    """Run tests with the 'favorites' dependencies group."""
    # The 'favorites' extra in pyproject.toml should now just list packages directly
    session.install(".[favorites,test]")
    session.run("pytest", "tests", "-n", "auto", "-m", "not tutorial")


@nox.session(name="tutorials", python="3.10")
def tutorials(session):
    """Execute markdown tutorials once to populate executed notebooks for docs."""
    # Install dev extras that include jupytext/nbclient etc.
    session.install(".[all,dev]")
    session.install("surya-ocr")
    session.install("easyocr")
    session.install("doclayout_yolo")
    # Run only tests marked as tutorial (no repetition across envs)
    workers = os.environ.get("NOTEBOOK_WORKERS", "10")
    session.run("pytest", "tests", "-m", "tutorial", "-n", workers)


# Optional: Add a test dependency group to pyproject.toml if needed
# [project.optional-dependencies]
# test = [
#     "pytest",
#     "pytest-cov", # Optional for coverage
# ]
