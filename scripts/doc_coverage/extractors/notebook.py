"""Extract code samples from Jupyter notebooks."""

import json
from pathlib import Path

from .base import CodeSample


def extract_from_notebook(path: Path) -> list[CodeSample]:
    """Extract code cells from a Jupyter notebook.

    Args:
        path: Path to the .ipynb file.

    Returns:
        List of CodeSample objects containing the extracted code.
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    samples = []

    cells = nb.get("cells", [])
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        # Handle both string and list formats for source
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Skip empty cells or magic/shell commands
        stripped = source.strip()
        if not stripped or stripped.startswith(("%", "!")):
            continue

        samples.append(
            CodeSample(
                source=source,
                file_path=str(path),
                location=f"cell_{i}",
            )
        )

    return samples
