from pathlib import Path

import pytest

# Conditionally import heavy dependencies; skip tests if unavailable in the environment.
jupytext = pytest.importorskip("jupytext")
nbformat = pytest.importorskip("nbformat")

# Directory that holds all markdown tutorials (adjust if layout changes)
TUTORIALS_DIR = Path(__file__).resolve().parent.parent / "docs" / "tutorials"

# Collect every *.md file in the tutorials directory (non-recursive)
MD_TUTORIALS = sorted(TUTORIALS_DIR.glob("*.md"))

pytestmark = pytest.mark.tutorial


@pytest.mark.parametrize("md_path", MD_TUTORIALS, ids=[p.stem for p in MD_TUTORIALS])
def test_tutorial_markdown_executes(md_path: Path):
    """Check that the markdown tutorial has been converted to a notebook with outputs.

    This test verifies that:
    1. A corresponding .ipynb file exists
    2. The notebook has been executed (contains outputs)

    It does NOT re-execute the notebook, as that should be done by 01-execute_notebooks.py
    """
    # Path where the executed notebook should exist
    ipynb_path = md_path.with_suffix(".ipynb")

    # Check that the notebook exists
    if not ipynb_path.exists():
        pytest.fail(f"Expected notebook not found: {ipynb_path}")

    # Read and verify the notebook has outputs
    try:
        with open(ipynb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        pytest.fail(f"Failed to read notebook {ipynb_path}: {e}")

    # Check that at least some cells have outputs (indicating execution)
    cells_with_output = 0
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.get("outputs"):
            cells_with_output += 1

    # If there are code cells but none have outputs, the notebook wasn't executed
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if code_cells and cells_with_output == 0:
        pytest.fail(
            f"Notebook {ipynb_path.name} exists but appears not to have been executed (no outputs found)"
        )

    # Optional: Check for execution errors in outputs
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    # Get error details
                    ename = output.get("ename", "Unknown")
                    evalue = output.get("evalue", "No error message")
                    pytest.fail(
                        f"Notebook {ipynb_path.name} has execution error in cell: {ename}: {evalue}"
                    )
