from __future__ import annotations

import ast
import re
from pathlib import Path

DOCS_ROOT = Path("docs")


def _section(markdown: str, heading: str) -> str:
    marker = f"### {heading}"
    start = markdown.index(marker)
    following_heading = markdown.find("\n### ", start + len(marker))
    if following_heading == -1:
        return markdown[start:]
    return markdown[start:following_heading]


def _python_blocks(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)


def test_landing_page_keeps_anchor_guides_recovery_pattern_visible():
    index = (DOCS_ROOT / "index.md").read_text()

    section = _section(index, "Recover Difficult Tables")
    assert "extract_table_guided" in section
    assert "extract_anchored_rows" in section
    assert 'cell_overlap="center"' in section

    blocks = _python_blocks(section)
    assert len(blocks) == 2
    for block in blocks:
        ast.parse(block)


def test_llms_txt_names_anchor_guides_recovery_pattern():
    llms = (DOCS_ROOT / "llms.txt").read_text()

    assert "page.extract_anchored_rows(anchors, side='right')" in llms
    assert "pdf.pages.extract_anchored_rows(anchors, side='right')" in llms
    assert "page.extract_table_guided(headers, row_anchors, header_anchor=...)" in llms
    assert "page.guides().from_headers_and_row_anchors" in llms
    assert "do not jump straight to OCR" in llms
    assert "cell_overlap='center'" in llms
