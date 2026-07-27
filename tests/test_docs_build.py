"""Tests for scripts/docs_build.py (executed-markdown docs pipeline)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("jupytext", reason="docs_build imports jupytext at load time")
pytest.importorskip("nbclient", reason="docs_build imports nbclient at load time")
PIL_Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def docs_build():
    script_path = REPO_ROOT / "scripts" / "docs_build.py"
    spec = importlib.util.spec_from_file_location("docs_build", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves string annotations via sys.modules[cls.__module__]
    sys.modules["docs_build"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def test_parse_frontmatter_extracts_keys_and_strips_block(docs_build):
    text = "---\nfixture: pdfs/01-practice.pdf\nthumbnail: 2\nskip: true\n---\n# Title\n\nBody.\n"
    meta, body = docs_build.parse_frontmatter(text)
    assert meta == {"fixture": "pdfs/01-practice.pdf", "thumbnail": 2, "skip": True}
    assert body == "# Title\n\nBody.\n"


def test_parse_frontmatter_without_block_returns_empty_meta(docs_build):
    text = "# Title\n\nNo frontmatter here.\n"
    meta, body = docs_build.parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_ignores_mid_document_rules(docs_build):
    text = "# Title\n\n---\nnot: frontmatter\n---\n"
    meta, body = docs_build.parse_frontmatter(text)
    assert meta == {}
    assert body == text


# ---------------------------------------------------------------------------
# Document model / tabs
# ---------------------------------------------------------------------------

TABBED_PAGE = """# Page

Intro text.

/// tab | First way

```python
a = 1
a
```

///

/// tab | Second way

```python
b = 2
```

///

Outro.
"""


def test_parse_document_structures_tabs_and_indices(docs_build):
    nodes = docs_build.parse_document(TABBED_PAGE)
    tabs = [n for n in nodes if isinstance(n, docs_build.TabNode)]
    assert [t.title for t in tabs] == ["First way", "Second way"]
    code = list(docs_build.iter_code_nodes(nodes))
    assert [c.index for c in code] == [0, 1]
    assert code[0].code == "a = 1\na"
    assert docs_build.count_code_nodes(nodes) == 2


def test_tab_round_trip_preserves_markers_in_executed_markdown(docs_build):
    nodes = docs_build.parse_document(TABBED_PAGE)
    out = docs_build.render_executed_markdown(nodes, {})
    assert "/// tab | First way" in out
    assert "/// tab | Second way" in out
    # each tab contributes one "/// tab |" open and one bare "///" close
    assert out.count("/// tab |") == 2
    assert len([ln for ln in out.splitlines() if ln.strip() == "///"]) == 2
    reparsed = docs_build.parse_document(out)
    retabs = [n for n in reparsed if isinstance(n, docs_build.TabNode)]
    assert [t.title for t in retabs] == ["First way", "Second way"]
    assert docs_build.count_code_nodes(reparsed) == 2


def test_outputs_injected_inside_the_right_tab(docs_build):
    nodes = docs_build.parse_document(TABBED_PAGE)
    rendered = docs_build.RenderedOutputs(snippets=["```output\n1\n```"])
    out = docs_build.render_executed_markdown(nodes, {0: rendered})
    first_tab = out.split("/// tab | Second way")[0]
    assert "```output\n1\n```" in first_tab
    second_tab = out.split("/// tab | Second way")[1]
    assert "```output" not in second_tab


def test_notebook_markdown_turns_tabs_into_bold_titles(docs_build):
    nodes = docs_build.parse_document(TABBED_PAGE)
    nb_md = docs_build.nodes_to_notebook_markdown(nodes)
    assert "**First way**" in nb_md
    assert "**Second way**" in nb_md
    assert "///" not in nb_md


def test_unclosed_tab_raises(docs_build):
    with pytest.raises(docs_build.DocsBuildError):
        docs_build.parse_document("/// tab | Oops\n\ntext\n")


def test_non_python_fences_stay_markdown(docs_build):
    body = "```bash\nls\n```\n\n```python\nx = 1\n```\n"
    nodes = docs_build.parse_document(body)
    assert docs_build.count_code_nodes(nodes) == 1
    md = [n for n in nodes if isinstance(n, docs_build.MarkdownNode)]
    assert any("```bash" in n.text for n in md)


# ---------------------------------------------------------------------------
# Asset hashing determinism
# ---------------------------------------------------------------------------


def _png_bytes(color):
    import io

    img = PIL_Image.new("RGB", (20, 10), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_save_png_asset_is_content_addressed_and_deterministic(docs_build, tmp_path):
    data = _png_bytes((255, 0, 0))
    name1 = docs_build.save_png_asset(data, tmp_path)
    name2 = docs_build.save_png_asset(data, tmp_path)
    assert name1 == name2
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert name1 == docs_build.asset_filename(docs_build.normalize_png(data))

    other = docs_build.save_png_asset(_png_bytes((0, 0, 255)), tmp_path)
    assert other != name1
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_normalize_png_strips_text_metadata(docs_build):
    import io

    from PIL import PngImagePlugin

    img = PIL_Image.new("RGB", (5, 5), (1, 2, 3))
    info = PngImagePlugin.PngInfo()
    info.add_text("CreationTime", "2026-07-27T00:00:00")
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    with_meta = buf.getvalue()

    buf2 = io.BytesIO()
    PIL_Image.new("RGB", (5, 5), (1, 2, 3)).save(buf2, format="PNG")
    without_meta = buf2.getvalue()

    assert with_meta != without_meta
    assert docs_build.normalize_png(with_meta) == docs_build.normalize_png(without_meta)


# ---------------------------------------------------------------------------
# End-to-end (tiny page, real kernel, no model downloads)
# ---------------------------------------------------------------------------

TINY_PAGE = """---
fixture: none
---

# Tiny page

```python
value = 40 + 2
print(f"value is {value}")
```

A bare expression:

```python
value
```
"""


@pytest.mark.tutorial
def test_end_to_end_build_cache_and_force(docs_build, tmp_path):
    src = tmp_path / "tiny.md"
    src.write_text(TINY_PAGE, encoding="utf-8")
    out_dir = tmp_path / "out"
    cache_file = tmp_path / "cache.json"

    result = docs_build.build_page(src, out_dir, cache_file=cache_file)
    assert result.status == "built"

    # Executed markdown with output blocks, no frontmatter
    md = (out_dir / "tiny.md").read_text(encoding="utf-8")
    assert "fixture:" not in md
    assert "```output\nvalue is 42\n```" in md
    assert "```output\n42\n```" in md

    # Notebook artifact: Colab badge + pip install + 2 code cells
    nb = nbformat.read(str(out_dir / "notebooks" / "tiny.ipynb"), as_version=4)
    assert nb.cells[0].cell_type == "markdown"
    assert "colab.research.google.com" in nb.cells[0].source
    assert nb.cells[1].cell_type == "code"
    assert nb.cells[1].source == '%pip install "natural-pdf[all]"'
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 3  # pip cell + two fences
    assert all(c.outputs == [] for c in code_cells)  # unexecuted

    # Cache: second run is a hit and outputs are unchanged
    before = {p: p.read_bytes() for p in sorted(out_dir.rglob("*")) if p.is_file()}
    cached = docs_build.build_page(src, out_dir, cache_file=cache_file)
    assert cached.status == "cached"
    after = {p: p.read_bytes() for p in sorted(out_dir.rglob("*")) if p.is_file()}
    assert before == after

    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert len(cache_data) == 1
    (entry,) = cache_data.values()
    assert entry["natural_pdf_version"] == docs_build.natural_pdf_version()

    # --force re-executes to byte-identical outputs
    forced = docs_build.build_page(src, out_dir, force=True, cache_file=cache_file)
    assert forced.status == "built"
    after_force = {p: p.read_bytes() for p in sorted(out_dir.rglob("*")) if p.is_file()}
    assert before == after_force


def test_cache_invalidated_by_source_change(docs_build, tmp_path, monkeypatch):
    """Cache logic without a kernel: monkeypatch execution."""
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cache_file = tmp_path / "cache.json"

    calls = []

    def fake_execute(nodes, **kwargs):
        calls.append(1)
        exec_md = docs_build.nodes_to_execution_markdown(nodes)
        from jupytext import jupytext as jt

        return jt.reads(exec_md, fmt="md")

    monkeypatch.setattr(docs_build, "execute_page_notebook", fake_execute)

    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "cached"
    assert len(calls) == 1

    src.write_text("# P\n\n```python\nx = 2\n```\n", encoding="utf-8")
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert len(calls) == 2

    # Deleting an output invalidates the cache entry too
    (out_dir / "page.md").unlink()
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert len(calls) == 3


def test_frontmatter_skip_passes_page_through(docs_build, tmp_path, monkeypatch):
    src = tmp_path / "skippy.md"
    src.write_text(
        "---\nskip: true\n---\n# Skipped\n\n```python\nraise SystemExit\n```\n",
        encoding="utf-8",
    )

    def boom(*args, **kwargs):
        raise AssertionError("skip: true page must not execute")

    monkeypatch.setattr(docs_build, "execute_page_notebook", boom)

    result = docs_build.build_page(src, tmp_path / "out", cache_file=tmp_path / "c.json")
    assert result.status == "skipped"
    md = (tmp_path / "out" / "skippy.md").read_text(encoding="utf-8")
    assert "skip: true" not in md
    assert "```python\nraise SystemExit\n```" in md
