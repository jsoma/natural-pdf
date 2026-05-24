"""Tests for the documentation notebook executor script."""

import importlib.util
from pathlib import Path

import nbformat


def load_executor_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01-execute_notebooks.py"
    spec = importlib.util.spec_from_file_location("execute_notebooks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_markdown_files_only_returns_executable_notebook_sources(tmp_path):
    executor = load_executor_module()
    docs_dir = tmp_path / "docs"

    for path in [
        docs_dir / "tutorials" / "01-intro.md",
        docs_dir / "quick-reference" / "index.md",
        docs_dir / "cookbook" / "guides.md",
        docs_dir / "getting-started" / "quickstart.md",
        docs_dir / "index.md",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Example\n", encoding="utf-8")

    found = executor.find_markdown_files(docs_dir, [])
    relative_paths = {path.relative_to(docs_dir).as_posix() for path in found}

    assert relative_paths == {
        "tutorials/01-intro.md",
        "quick-reference/index.md",
    }


def test_normalize_notebook_metadata_promotes_truthy_skip_to_nbclient_tag():
    executor = load_executor_module()
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("x = 1", metadata={"skip": True}),
            nbformat.v4.new_code_cell("y = 2", metadata={"skip": False}),
        ]
    )

    executor.normalize_notebook_metadata(notebook)

    assert "skip-execution" in notebook.cells[0].metadata["tags"]
    assert "tags" not in notebook.cells[1].metadata
