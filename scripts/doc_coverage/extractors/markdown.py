"""Extract code samples from Markdown files."""

import re
from pathlib import Path

from .base import CodeSample


def extract_from_markdown(path: Path) -> list[CodeSample]:
    """Extract Python code blocks from a Markdown file.

    Args:
        path: Path to the Markdown file.

    Returns:
        List of CodeSample objects containing the extracted code.
    """
    content = path.read_text(encoding="utf-8")

    # Match ```python or ```py code blocks
    pattern = r"```(?:python|py)\n(.*?)```"
    samples = []

    for i, match in enumerate(re.finditer(pattern, content, re.DOTALL)):
        source = match.group(1)
        # Calculate line number of the match
        line_number = content[: match.start()].count("\n") + 1

        samples.append(
            CodeSample(
                source=source,
                file_path=str(path),
                location=f"line_{line_number}",
            )
        )

    return samples
