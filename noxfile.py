import sys
import os

import nox

# Ensure nox uses the same Python version you are developing with or whichever is appropriate
# Make sure this Python version has nox installed (`pip install nox`)
# You can specify multiple Python versions to test against, e.g., ["3.9", "3.10", "3.11"]
nox.options.sessions = ["lint", "test_core", "test_extras", "test_all", "test_favorites"]
nox.options.reuse_existing_virtualenvs = True  # Faster runs by reusing environments
nox.options.default_venv_backend = "uv"  # Use uv for faster venv creation and package installation

PYTHON_VERSIONS = (
    ["3.9", "3.10", "3.11"] if sys.platform != "darwin" else ["3.9", "3.10", "3.11"]
)  # Add more as needed

# Updated to match pyproject.toml optional dependencies
OPTIONAL_DEPS = [
    "viewer", 
    "easyocr", 
    "paddle", 
    "layout_yolo", 
    "surya", 
    "doctr",
    "docling",
    "llm",
    "search",
    "deskew",
    "core-ml",
    "ocr-export"
]


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    """Run linters."""
    session.install("black", "isort", "mypy")
    session.run("black", "--check", ".")
    session.run("isort", "--check-only", ".")
    # Consider adding mypy checks if types are consistently added
    # session.run("mypy", "src", "tests") # Adjust paths as needed


@nox.session(python=PYTHON_VERSIONS)
def test_core(session):
    """Run tests with only core dependencies."""
    session.install(".[test]")  # Assuming you add pytest etc to a [test] extra or install directly
    
    # Run both the original tests and new test structure
    # Prioritize the new structure but keep backward compatibility
    if os.path.exists("tests/test_core"):
        session.run("pytest", "tests/test_core", "tests/test_loading_original.py", "tests/exporters")
    else:
        session.run("pytest", "tests")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("extra", OPTIONAL_DEPS)
def test_extras(session, extra):
    """Run tests with each optional dependency group."""
    # Skip paddle on macOS for now due to potential build issues
    if extra == "paddle" and sys.platform == "darwin":
        session.skip("PaddleOCR installation can be complex on macOS")

    # Skip surya on Python 3.9 due to dependency incompatibility
    if extra == "surya" and session.python == "3.9":
        session.skip("Surya-OCR dependency uses syntax incompatible with Python 3.9")
        
    # Skip docling on Python 3.9 if it has the same dependency issue as surya
    if extra == "docling" and session.python == "3.9":
        session.skip("Docling dependency may use syntax incompatible with Python 3.9")

    # Install the package with the specified extra
    # Also include 'test' extras if your test runners/libs are there
    session.install(f".[{extra},test]")
    session.run("pytest", "tests")


@nox.session(python=PYTHON_VERSIONS)
def test_all(session):
    """Run tests with all optional dependencies."""
    # Skip this entire session on Python 3.9 because surya is incompatible
    if session.python == "3.9":
        session.skip("Cannot run 'all' extras on Python 3.9 due to surya-ocr incompatibility.")

    # Skip paddle on macOS for now
    if sys.platform == "darwin":
        # Install with 'all' which should include everything except paddle on macOS
        session.install(".[all,test]")
    else:
        session.install(".[all,test]")

    session.run("pytest", "tests")


@nox.session(python=PYTHON_VERSIONS)
def test_favorites(session):
    """Run tests with the 'favorites' dependencies group."""
    # The favorites group includes the most commonly used dependencies
    session.install(".[favorites,test]")
    session.run("pytest", "tests")


# Optional: Add a test dependency group to pyproject.toml if needed
# [project.optional-dependencies]
# test = [
#     "pytest",
#     "pytest-cov", # Optional for coverage
# ]
