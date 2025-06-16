from pathlib import Path
import pytest

# Conditionally import heavy dependencies; skip tests if unavailable in the environment.
jupytext = pytest.importorskip("jupytext")
nbformat = pytest.importorskip("nbformat")
nbclient_mod = pytest.importorskip("nbclient")
# Import NotebookClient from nbclient root, but CellExecutionError lives in nbclient.exceptions
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Directory that holds all markdown tutorials (adjust if layout changes)
TUTORIALS_DIR = Path(__file__).resolve().parent.parent / "docs" / "tutorials"

# Collect every *.md file in the tutorials directory (non-recursive)
MD_TUTORIALS = sorted(TUTORIALS_DIR.glob("*.md"))

pytestmark = pytest.mark.tutorial

@pytest.mark.parametrize("md_path", MD_TUTORIALS, ids=[p.stem for p in MD_TUTORIALS])
def test_tutorial_markdown_executes(md_path: Path):
    """Convert the markdown tutorial to a notebook, execute it, and save outputs.

    The test will fail if any cell errors, mirroring the behaviour of nbclient.
    The executed notebook is written back to the same location with the .ipynb
    extension so it can be published in docs builds.
    """
    # Read markdown and convert to an nbformat.NotebookNode via jupytext
    md_text = md_path.read_text(encoding="utf-8")
    notebook = jupytext.reads(md_text, fmt="md")

    # Path where we will write the executed notebook
    ipynb_path = md_path.with_suffix(".ipynb")

    # Execute the notebook in-memory
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",  # Use default Python kernel available in CI/local env
        resources={"metadata": {"path": str(md_path.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        # Re-raise with a clearer assertion message for Pytest output
        pytest.fail(f"Execution failed for {md_path.name}: {exc}")

    # Persist the executed notebook so rendered outputs are available for docs
    ipynb_path.write_text(nbformat.writes(notebook), encoding="utf-8") 