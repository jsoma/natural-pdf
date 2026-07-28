"""Tests for scripts/generate_api_reference.py.

Runs the Griffe-based fragment generator into temporary directories and checks
that its output matches the include markers in the authored docs/api pages,
contains known symbols, is byte-for-byte deterministic, and carries no leftover
mkdocstrings syntax.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("griffe")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "generate_api_reference.py"
API_PAGES = (
    REPO_ROOT / "docs" / "api" / "index.md",
    REPO_ROOT / "docs" / "api" / "text-extraction.md",
)
MARKER_RE = re.compile(r"<!--\s*npdf-api:include\s+id=([A-Za-z0-9][A-Za-z0-9-]*)\s*-->")


def _run_generator(out_dir: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"generator failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _referenced_ids() -> set[str]:
    ids: set[str] = set()
    for page in API_PAGES:
        ids.update(MARKER_RE.findall(page.read_text(encoding="utf-8")))
    return ids


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, Path]:
    """Run the generator twice into separate directories (also used for determinism)."""
    first = tmp_path_factory.mktemp("api-fragments-1")
    second = tmp_path_factory.mktemp("api-fragments-2")
    _run_generator(first)
    _run_generator(second)
    return first, second


def test_docs_pages_reference_at_least_the_nine_fragments():
    ids = _referenced_ids()
    assert len(ids) == 9, f"expected 9 include markers across docs/api pages, got {sorted(ids)}"


def test_every_referenced_include_id_has_a_fragment(generated):
    first, _ = generated
    generated_ids = {path.stem for path in first.glob("*.md")}
    missing = _referenced_ids() - generated_ids
    assert (
        not missing
    ), f"docs/api pages reference ids with no generated fragment: {sorted(missing)}"


def test_known_symbols_are_documented(generated):
    first, _ = generated
    root_fragment = (first / "natural-pdf.md").read_text(encoding="utf-8")
    assert '<a id="natural_pdf.PDF"></a>' in root_fragment
    assert "### `PDF`" in root_fragment
    assert "extract_text" in root_fragment

    layout_fragment = (first / "text-layout-options.md").read_text(encoding="utf-8")
    assert "### `TextLayoutOptions`" in layout_fragment


def test_generation_is_deterministic(generated):
    first, second = generated
    first_files = sorted(path.name for path in first.glob("*.md"))
    second_files = sorted(path.name for path in second.glob("*.md"))
    assert first_files == second_files
    assert first_files, "generator produced no fragments"
    for name in first_files:
        assert (first / name).read_bytes() == (
            second / name
        ).read_bytes(), f"fragment {name} differs between two runs"


def test_fragments_contain_no_mkdocstrings_syntax(generated):
    first, _ = generated
    for path in sorted(first.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not line.lstrip().startswith(
                ":::"
            ), f"mkdocstrings directive leaked into {path.name}: {line!r}"
