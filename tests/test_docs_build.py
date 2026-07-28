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


def test_parse_frontmatter_rejects_invalid_yaml(docs_build):
    text = "---\nskip: [unterminated\n---\n# Title\n"
    with pytest.raises(docs_build.DocsBuildError, match="Invalid YAML frontmatter"):
        docs_build.parse_frontmatter(text)


def test_parse_frontmatter_rejects_non_mapping(docs_build):
    with pytest.raises(docs_build.DocsBuildError, match="must be a mapping"):
        docs_build.parse_frontmatter("---\n- skip\n- true\n---\n# Title\n")


def test_parse_frontmatter_accepts_closing_delimiter_at_eof(docs_build):
    meta, body = docs_build.parse_frontmatter("---\ntier: nightly\n---")
    assert meta == {"tier": "nightly"}
    assert body == ""


def test_parse_frontmatter_rejects_missing_closing_delimiter(docs_build):
    with pytest.raises(docs_build.DocsBuildError, match="Unclosed YAML frontmatter"):
        docs_build.parse_frontmatter("---\nskip: true\n# no closing delimiter\n")


def test_parse_frontmatter_preserves_crlf_body(docs_build):
    meta, body = docs_build.parse_frontmatter("---\r\ntier: fast\r\n---\r\n# Title\r\n")
    assert meta == {"tier": "fast"}
    assert body == "# Title\r\n"


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
    assert nb.cells[1].source == docs_build.PIP_INSTALL_CELL
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


def test_cache_invalidated_by_runtime_or_colab_context(docs_build, tmp_path, monkeypatch):
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cache_file = tmp_path / "cache.json"
    calls = []
    _install_fake_execute(docs_build, monkeypatch, calls=calls)
    fingerprint = ["runtime-a"]
    monkeypatch.setattr(docs_build, "execution_fingerprint", lambda: fingerprint[0])

    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "cached"

    fingerprint[0] = "runtime-b"
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert (
        docs_build.build_page(src, out_dir, cache_file=cache_file, colab_prefix="learn").status
        == "built"
    )
    assert len(calls) == 3


def test_referenced_fixture_fingerprint_tracks_only_page_inputs(docs_build, tmp_path, monkeypatch):
    monkeypatch.setattr(docs_build, "REPO_ROOT", tmp_path)
    fixtures = tmp_path / "pdfs"
    fixtures.mkdir()
    referenced = fixtures / "used.pdf"
    referenced.write_bytes(b"version one")
    unrelated = fixtures / "unrelated.pdf"
    unrelated.write_bytes(b"unrelated one")
    page = '# P\n\npdf = PDF("pdfs/used.pdf")\n'

    first = docs_build.referenced_fixture_fingerprint(page)
    unrelated.write_bytes(b"unrelated two")
    assert docs_build.referenced_fixture_fingerprint(page) == first

    referenced.write_bytes(b"version two")
    assert docs_build.referenced_fixture_fingerprint(page) != first


def test_cache_invalidated_by_referenced_fixture_change(docs_build, tmp_path, monkeypatch):
    src = tmp_path / "page.md"
    src.write_text('# P\n\n```python\nPDF("pdfs/example.pdf")\n```\n', encoding="utf-8")
    out_dir = tmp_path / "out"
    cache_file = tmp_path / "cache.json"
    calls = []
    _install_fake_execute(docs_build, monkeypatch, calls=calls)
    fixture_fingerprint = ["fixture-a"]
    monkeypatch.setattr(
        docs_build,
        "referenced_fixture_fingerprint",
        lambda text: fixture_fingerprint[0],
    )

    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "cached"
    fixture_fingerprint[0] = "fixture-b"
    assert docs_build.build_page(src, out_dir, cache_file=cache_file).status == "built"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Fake execution helper (no kernel): jupytext-parse the page, normalize
# metadata like the real path does, then attach fabricated outputs per cell.
# ---------------------------------------------------------------------------


def _install_fake_execute(docs_build, monkeypatch, outputs_for_cell=None, calls=None):
    from jupytext import jupytext as jt

    def fake_execute(nodes, **kwargs):
        if calls is not None:
            calls.append(1)
        nb = jt.reads(docs_build.nodes_to_execution_markdown(nodes), fmt="md")
        docs_build.normalize_notebook_metadata(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        for i, cell in enumerate(code_cells):
            cell["outputs"] = outputs_for_cell(i) if outputs_for_cell else []
        return nb

    monkeypatch.setattr(docs_build, "execute_page_notebook", fake_execute)


def _stream_output(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def _png_output(color):
    import base64

    return {
        "output_type": "display_data",
        "data": {"image/png": base64.b64encode(_png_bytes(color)).decode("ascii")},
    }


# ---------------------------------------------------------------------------
# hide-output fence attribute
# ---------------------------------------------------------------------------


def test_normalize_metadata_maps_hide_output_to_tag(docs_build):
    from jupytext import jupytext as jt

    md = (
        "```python hide-output\nprint(1)\n```\n\n"
        "```python hide-output=true\nprint(2)\n```\n\n"
        "```python\nprint(3)\n```\n"
    )
    nb = jt.reads(md, fmt="md")
    docs_build.normalize_notebook_metadata(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert docs_build.cell_hides_output(code[0])  # bare attribute
    assert docs_build.cell_hides_output(code[1])  # =true form
    assert not docs_build.cell_hides_output(code[2])


def test_hide_output_executes_but_omits_outputs(docs_build, tmp_path, monkeypatch):
    src = tmp_path / "page.md"
    src.write_text(
        "# Page\n\n"
        "```python hide-output\nprint('secret')\n```\n\n"
        "```python\nprint('visible')\n```\n",
        encoding="utf-8",
    )
    _install_fake_execute(
        docs_build,
        monkeypatch,
        outputs_for_cell=lambda i: [_stream_output(f"out{i}\n")],
    )
    docs_build.build_page(src, tmp_path / "out", cache_file=tmp_path / "c.json")
    md = (tmp_path / "out" / "page.md").read_text(encoding="utf-8")
    assert "out0" not in md  # executed, but output omitted
    assert "```output\nout1\n```" in md
    # attribute is stripped from the rendered fence
    assert "hide-output" not in md
    assert "```python\nprint('secret')\n```" in md


# ---------------------------------------------------------------------------
# Alt text for images
# ---------------------------------------------------------------------------


def test_render_cell_outputs_numbers_alt_for_multiple_images(docs_build, tmp_path):
    cell = {"outputs": [_png_output((255, 0, 0)), _png_output((0, 255, 0))]}
    rendered = docs_build.render_cell_outputs(cell, tmp_path, "assets/p", alt="Chart")
    assert rendered.snippets[0].startswith("![Chart](assets/p/")
    assert rendered.snippets[1].startswith("![Chart (2)](assets/p/")


def test_render_cell_outputs_empty_alt_stays_empty(docs_build, tmp_path):
    cell = {"outputs": [_png_output((255, 0, 0)), _png_output((0, 255, 0))]}
    rendered = docs_build.render_cell_outputs(cell, tmp_path, "assets/p", alt="")
    assert rendered.snippets[0].startswith("![](")
    assert rendered.snippets[1].startswith("![](")  # no " (2)" on empty alt


def test_compute_default_alts_tracks_nearest_heading(docs_build):
    body = (
        "```python\nbefore_any_heading = 1\n```\n\n"
        "# Title\n\n"
        "## Rendering the page\n\n"
        "```python\nimg\n```\n\n"
        "Text between.\n\n"
        "```python\nimg2\n```\n\n"
        "### Another view\n\n"
        "```bash\n# not a heading (inside a fence)\nls\n```\n\n"
        "```python\nimg3\n```\n"
    )
    nodes = docs_build.parse_document(body)
    alts = docs_build.compute_default_alts(nodes)
    assert alts[0] == ""  # no preceding heading
    assert alts[1] == "Rendering the page"
    assert alts[2] == "Rendering the page"
    assert alts[3] == "Another view"


def test_fence_alt_attribute_overrides_heading_default(docs_build):
    nodes = docs_build.parse_document(
        '## Section\n\n```python alt="Custom alt"\nimg\n```\n\n```python\nimg2\n```\n'
    )
    alts = docs_build.compute_default_alts(nodes)
    code = list(docs_build.iter_code_nodes(nodes))
    assert docs_build.fence_alt(code[0], alts) == "Custom alt"
    assert docs_build.fence_alt(code[1], alts) == "Section"


def test_build_page_applies_heading_alt_and_alt_attribute(docs_build, tmp_path, monkeypatch):
    src = tmp_path / "page.md"
    src.write_text(
        "## Rendering the page\n\n"
        "```python\nimg\n```\n\n"
        '```python alt="Custom alt"\nimg\n```\n',
        encoding="utf-8",
    )
    _install_fake_execute(
        docs_build,
        monkeypatch,
        outputs_for_cell=lambda i: [
            _png_output((255, 0, 0)) if i == 0 else _png_output((0, 0, 255))
        ],
    )
    docs_build.build_page(src, tmp_path / "out", cache_file=tmp_path / "c.json")
    md = (tmp_path / "out" / "page.md").read_text(encoding="utf-8")
    assert "![Rendering the page](assets/page/" in md
    assert "![Custom alt](assets/page/" in md
    assert 'alt="Custom alt"' not in md  # attribute stripped from the fence


# ---------------------------------------------------------------------------
# Pandas <style scoped> stripping
# ---------------------------------------------------------------------------


def test_style_scoped_block_is_stripped_from_html_outputs(docs_build, tmp_path):
    html = (
        "<div>\n<style scoped>\n    .dataframe tbody tr th { vertical-align: top; }\n"
        "</style>\n<table><tr><td>1</td></tr></table>\n</div>"
    )
    cell = {
        "outputs": [
            {
                "output_type": "execute_result",
                "data": {"text/html": html, "text/plain": "a df"},
            }
        ]
    }
    rendered = docs_build.render_cell_outputs(cell, tmp_path, "assets/p")
    assert len(rendered.snippets) == 1
    assert "<style scoped>" not in rendered.snippets[0]
    assert ".dataframe" not in rendered.snippets[0]
    assert "<table><tr><td>1</td></tr></table>" in rendered.snippets[0]


# ---------------------------------------------------------------------------
# Trailing whitespace trim inside output blocks
# ---------------------------------------------------------------------------


def test_output_block_lines_are_right_trimmed_but_aligned(docs_build, tmp_path):
    cell = {"outputs": [_stream_output("col_a  col_b   \n1      2       \n")]}
    rendered = docs_build.render_cell_outputs(cell, tmp_path, "assets/p")
    assert rendered.snippets == ["```output\ncol_a  col_b\n1      2\n```"]


# ---------------------------------------------------------------------------
# Colab fixture path rewrite (.ipynb artifact only)
# ---------------------------------------------------------------------------


def test_notebook_rewrites_fixture_paths_but_markdown_keeps_them(docs_build, tmp_path):
    body = (
        "# T\n\n"
        "```python\n"
        'pdf = PDF("pdfs/01-practice.pdf")\n'
        "other = PDF('pdfs/Atlanta_Public_Schools_GA_sample.pdf')\n"
        'remote = PDF("https://example.com/some.pdf")\n'
        "```\n"
    )
    nodes = docs_build.parse_document(body)
    nb = docs_build.build_notebook_artifact(nodes, tmp_path / "notebooks" / "t.ipynb")
    code = [c for c in nb.cells if c.cell_type == "code"]
    # code[0] is the pip-install cell
    source = code[1].source
    assert (
        '"https://raw.githubusercontent.com/jsoma/natural-pdf/main/pdfs/01-practice.pdf"' in source
    )
    assert (
        "'https://raw.githubusercontent.com/jsoma/natural-pdf/main/pdfs/"
        "Atlanta_Public_Schools_GA_sample.pdf'" in source
    )
    assert '"pdfs/' not in source and "'pdfs/" not in source
    assert 'PDF("https://example.com/some.pdf")' in source  # URL passes through

    # Executed markdown keeps the local path
    md = docs_build.render_executed_markdown(nodes, {})
    assert 'PDF("pdfs/01-practice.pdf")' in md
    assert "raw.githubusercontent.com" not in md


# ---------------------------------------------------------------------------
# Kernel lifecycle: timeout enforcement + no leaked kernels
# ---------------------------------------------------------------------------


@pytest.mark.tutorial
def test_timeout_enforced_and_kernel_shut_down(docs_build, tmp_path, monkeypatch):
    import time as _time

    kms = []
    orig = docs_build._DocsNotebookClient.create_kernel_manager

    def capture(self):
        km = orig(self)
        kms.append(km)
        return km

    monkeypatch.setattr(docs_build._DocsNotebookClient, "create_kernel_manager", capture)

    # A cell sleeping far past the timeout must fail the page promptly...
    slow = tmp_path / "slow.md"
    slow.write_text("# Slow\n\n```python\nimport time\ntime.sleep(60)\n```\n", encoding="utf-8")
    start = _time.monotonic()
    with pytest.raises(docs_build.DocsBuildError, match="timed out"):
        docs_build.build_page(slow, tmp_path / "out", timeout=2, cache_file=tmp_path / "c.json")
    assert _time.monotonic() - start < 40  # enforced, not waiting out the sleep

    # ...and the success path must also leave no kernel behind.
    fast = tmp_path / "fast.md"
    fast.write_text("# Fast\n\n```python\nprint('hi')\n```\n", encoding="utf-8")
    result = docs_build.build_page(fast, tmp_path / "out2", cache_file=tmp_path / "c.json")
    assert result.status == "built"

    assert len(kms) == 2
    assert all(not km.has_kernel for km in kms)  # no leaked kernel processes


# ---------------------------------------------------------------------------
# Output planning: tree mirroring, overwrite refusal, per-destination cache
# ---------------------------------------------------------------------------


def test_directory_build_mirrors_source_tree(docs_build, tmp_path, monkeypatch):
    root = tmp_path / "docsrc"
    (root / "solve").mkdir(parents=True)
    (root / "learn").mkdir()
    (root / "solve" / "index.md").write_text("# Solve\n\n```python\nx = 1\n```\n", encoding="utf-8")
    (root / "learn" / "index.md").write_text("# Learn\n\n```python\ny = 2\n```\n", encoding="utf-8")

    _install_fake_execute(
        docs_build, monkeypatch, outputs_for_cell=lambda i: [_png_output((9, 9, 9))]
    )
    out = tmp_path / "out"
    for src in docs_build.collect_sources(root):
        docs_build.build_page(src, out, cache_file=tmp_path / "c.json", source_root=root)

    solve = (out / "solve" / "index.md").read_text(encoding="utf-8")
    learn = (out / "learn" / "index.md").read_text(encoding="utf-8")
    assert "# Solve" in solve and "# Learn" in learn  # no overwrite
    assert (out / "solve" / "notebooks" / "index.ipynb").exists()
    assert (out / "learn" / "notebooks" / "index.ipynb").exists()
    # asset links stay relative to the page (resolve from <out>/solve/)
    assert "](assets/index/" in solve
    assert list((out / "solve" / "assets" / "index").glob("*.png"))
    assert list((out / "learn" / "assets" / "index").glob("*.png"))


def test_page_outside_source_root_is_an_error(docs_build, tmp_path, monkeypatch):
    _install_fake_execute(docs_build, monkeypatch)
    src = tmp_path / "page.md"
    src.write_text("# P\n", encoding="utf-8")
    with pytest.raises(docs_build.DocsBuildError, match="source root"):
        docs_build.build_page(
            src, tmp_path / "out", cache_file=tmp_path / "c.json", source_root=tmp_path / "other"
        )


def test_refuses_to_overwrite_authored_page(docs_build, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("must refuse before executing anything")

    monkeypatch.setattr(docs_build, "execute_page_notebook", boom)
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    with pytest.raises(docs_build.DocsBuildError, match="overwrite"):
        docs_build.build_page(src, tmp_path, cache_file=tmp_path / "c.json")
    assert src.read_text(encoding="utf-8").startswith("# P")  # untouched


def test_cache_is_per_destination(docs_build, tmp_path, monkeypatch):
    calls = []
    _install_fake_execute(docs_build, monkeypatch, calls=calls)
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    cache_file = tmp_path / "c.json"

    assert docs_build.build_page(src, tmp_path / "out1", cache_file=cache_file).status == "built"
    # Same page, different destination: must build (and write), not report cached
    second = docs_build.build_page(src, tmp_path / "out2", cache_file=cache_file)
    assert second.status == "built"
    assert (tmp_path / "out2" / "page.md").exists()
    # Original destination is still a cache hit
    assert docs_build.build_page(src, tmp_path / "out1", cache_file=cache_file).status == "cached"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Output hygiene: --out inside the source tree, stale thumbnails
# ---------------------------------------------------------------------------


def test_directory_build_rejects_out_inside_source(docs_build, tmp_path, capsys):
    root = tmp_path / "docsrc"
    root.mkdir()
    (root / "page.md").write_text("# P\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        docs_build.main([str(root), "--out", str(root / "generated"), "--no-execute"])
    assert excinfo.value.code != 0
    assert "inside the source directory" in capsys.readouterr().err
    # --out equal to the source root is refused too
    with pytest.raises(SystemExit):
        docs_build.main([str(root), "--out", str(root), "--no-execute"])
    # Single-file builds keep working with any out dir
    assert (
        docs_build.main(
            [
                str(root / "page.md"),
                "--out",
                str(tmp_path / "single-out"),
                "--no-execute",
                "--cache-file",
                str(tmp_path / "c.json"),
            ]
        )
        == 0
    )
    assert (tmp_path / "single-out" / "page.md").exists()


def test_collect_sources_excludes_out_dir(docs_build, tmp_path):
    root = tmp_path / "docsrc"
    (root / "generated").mkdir(parents=True)
    (root / "page.md").write_text("# P\n", encoding="utf-8")
    (root / "generated" / "page.md").write_text("# Old output\n", encoding="utf-8")

    with_exclude = docs_build.collect_sources(root, exclude_dir=root / "generated")
    assert with_exclude == [root / "page.md"]
    # Backward compatible: no exclude_dir discovers everything, as before
    assert len(docs_build.collect_sources(root)) == 2


def test_stale_thumbnail_deleted_when_frontmatter_dropped(docs_build, tmp_path, monkeypatch):
    _install_fake_execute(
        docs_build, monkeypatch, outputs_for_cell=lambda i: [_png_output((7, 7, 7))]
    )
    src = tmp_path / "page.md"
    out = tmp_path / "out"
    cache_file = tmp_path / "c.json"
    thumb = out / "assets" / "page-thumb.png"

    src.write_text("---\nthumbnail: 1\n---\n# P\n\n```python\nimg\n```\n", encoding="utf-8")
    result = docs_build.build_page(src, out, cache_file=cache_file)
    assert thumb.exists()
    assert thumb in result.outputs

    # Author removes `thumbnail:` — the stale artifact must go away.
    src.write_text("# P\n\n```python\nimg\n```\n", encoding="utf-8")
    result = docs_build.build_page(src, out, cache_file=cache_file)
    assert not thumb.exists()
    assert thumb not in result.outputs


def test_skip_page_removes_all_prior_execution_artifacts(docs_build, tmp_path, monkeypatch):
    _install_fake_execute(
        docs_build, monkeypatch, outputs_for_cell=lambda i: [_png_output((7, 7, 7))]
    )
    src = tmp_path / "page.md"
    out = tmp_path / "out"
    cache_file = tmp_path / "c.json"
    src.write_text("---\nthumbnail: 1\n---\n# P\n\n```python\nimg\n```\n", encoding="utf-8")
    docs_build.build_page(src, out, cache_file=cache_file)
    assert (out / "assets" / "page").exists()
    assert (out / "assets" / "page-thumb.png").exists()

    src.write_text("---\nskip: true\n---\n# P\n\n```python\nimg\n```\n", encoding="utf-8")
    result = docs_build.build_page(src, out, cache_file=cache_file)
    assert result.status == "skipped"
    assert not (out / "assets" / "page").exists()
    assert not (out / "assets" / "page-thumb.png").exists()
    assert docs_build._cache_key(src, out) not in docs_build.load_cache(cache_file)


def test_stale_cache_manifest_prunes_deleted_page_outputs(docs_build, tmp_path):
    source_root = tmp_path / "docs"
    out = tmp_path / "generated"
    cache_file = tmp_path / "cache.json"
    source_root.mkdir()
    keep = source_root / "keep.md"
    keep.write_text("# Keep\n", encoding="utf-8")
    deleted = source_root / "deleted.md"

    outputs = {}
    for page in ("keep", "deleted"):
        (out / f"{page}.md").parent.mkdir(parents=True, exist_ok=True)
        (out / f"{page}.md").write_text("generated", encoding="utf-8")
        (out / "notebooks").mkdir(exist_ok=True)
        (out / "notebooks" / f"{page}.ipynb").write_text("{}", encoding="utf-8")
        (out / "assets" / page).mkdir(parents=True, exist_ok=True)
        (out / "assets" / page / "image.png").write_bytes(b"png")
        (out / "assets" / f"{page}-thumb.png").write_bytes(b"thumb")
        outputs[page] = [
            out / f"{page}.md",
            out / "notebooks" / f"{page}.ipynb",
            out / "assets" / page / "image.png",
            out / "assets" / f"{page}-thumb.png",
        ]

    unrelated = out / "unrecorded.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    cache = {
        docs_build._cache_key(keep, out): {"outputs": [str(path) for path in outputs["keep"]]},
        docs_build._cache_key(deleted, out): {
            "outputs": [str(path) for path in outputs["deleted"]]
        },
    }
    docs_build.save_cache(cache_file, cache)

    removed = docs_build.prune_stale_cache_entries(cache_file, out, source_root, [keep])
    assert removed
    assert all(path.exists() for path in outputs["keep"])
    assert all(not path.exists() for path in outputs["deleted"])
    assert unrelated.exists()
    assert not (out / "assets" / "deleted").exists()
    assert set(docs_build.load_cache(cache_file)) == {docs_build._cache_key(keep, out)}


# ---------------------------------------------------------------------------
# Skipped fences in the exported notebook
# ---------------------------------------------------------------------------


def test_skip_fence_tagged_skip_execution_in_notebook(docs_build, tmp_path):
    body = (
        "# T\n\n"
        "```python {.skip-execution}\nclient = make_client()\n```\n\n"
        "```python {.skip-execution}\n# my own explanation\nx = 1\n```\n\n"
        "```python\ny = 2\n```\n"
    )
    nodes = docs_build.parse_document(body)
    nb = docs_build.build_notebook_artifact(nodes, tmp_path / "notebooks" / "t.ipynb")
    code = [c for c in nb.cells if c.cell_type == "code"]
    # code[0] is the pip-install cell
    assert "skip-execution" not in code[0].get("metadata", {}).get("tags", [])

    skip_cell = code[1]
    assert "skip-execution" in skip_cell.metadata["tags"]
    assert "skip" not in skip_cell.metadata
    assert skip_cell.source.splitlines()[0] == docs_build.SKIP_CELL_COMMENT
    assert "client = make_client()" in skip_cell.source

    # A skip cell that already opens with a comment only gets the tag.
    commented = code[2]
    assert "skip-execution" in commented.metadata["tags"]
    assert commented.source.splitlines()[0] == "# my own explanation"
    assert docs_build.SKIP_CELL_COMMENT not in commented.source

    plain = code[3]
    assert "skip-execution" not in plain.get("metadata", {}).get("tags", [])
    assert docs_build.SKIP_CELL_COMMENT not in plain.source


# ---------------------------------------------------------------------------
# Colab badge URL composition
# ---------------------------------------------------------------------------


def test_colab_url_defaults_and_options(docs_build, tmp_path):
    out = tmp_path / "out"
    nb_path = out / "learn" / "notebooks" / "x.ipynb"
    assert docs_build.colab_url(nb_path, out_root=out) == (
        "https://colab.research.google.com/github/jsoma/natural-pdf/"
        "blob/gh-pages/learn/notebooks/x.ipynb"
    )
    assert docs_build.colab_url(nb_path, ref="main", prefix="site/docs", out_root=out) == (
        "https://colab.research.google.com/github/jsoma/natural-pdf/"
        "blob/main/site/docs/learn/notebooks/x.ipynb"
    )
    # No out_root: falls back to notebooks/<name>
    assert docs_build.colab_url(nb_path).endswith("/blob/gh-pages/notebooks/x.ipynb")


def test_build_page_composes_badge_from_colab_options(docs_build, tmp_path, monkeypatch):
    _install_fake_execute(docs_build, monkeypatch)
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    out = tmp_path / "out"

    docs_build.build_page(
        src, out, cache_file=tmp_path / "c1.json", colab_ref="v1.2", colab_prefix="dl"
    )
    nb = nbformat.read(str(out / "notebooks" / "page.ipynb"), as_version=4)
    assert "blob/v1.2/dl/notebooks/page.ipynb)" in nb.cells[0].source

    # Default (no options): gh-pages ref, output-tree-relative path, no prefix
    out2 = tmp_path / "out2"
    docs_build.build_page(src, out2, cache_file=tmp_path / "c2.json")
    nb2 = nbformat.read(str(out2 / "notebooks" / "page.ipynb"), as_version=4)
    assert "blob/gh-pages/notebooks/page.ipynb)" in nb2.cells[0].source


def test_cli_colab_options_reach_the_badge(docs_build, tmp_path):
    src = tmp_path / "page.md"
    src.write_text("# P\n\n```python\nx = 1\n```\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = docs_build.main(
        [
            str(src),
            "--out",
            str(out),
            "--no-execute",
            "--cache-file",
            str(tmp_path / "c.json"),
            "--colab-ref",
            "release",
            "--colab-prefix",
            "notebooks-root",
        ]
    )
    assert rc == 0
    nb = nbformat.read(str(out / "notebooks" / "page.ipynb"), as_version=4)
    assert "blob/release/notebooks-root/notebooks/page.ipynb)" in nb.cells[0].source


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


def test_starlight_asides_do_not_break_jupytext_parsing(docs_build):
    """Column-0 ::: aside syntax must not make jupytext sniff the page as
    pandoc markdown (which collapses everything into one cell and loses
    every code fence). Regression test for the md -> md:markdown fmt pin."""
    import jupytext

    page = (
        ":::caution[Model download]\n"
        "This downloads a model on first use.\n"
        ":::\n"
        "\n"
        "Some prose.\n"
        "\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )
    nb = jupytext.reads(page, fmt="md:markdown")
    assert sum(1 for c in nb.cells if c.cell_type == "code") == 1
    # the exact failure mode of the unpinned "md" format:
    sniffed = jupytext.reads(page, fmt="md")
    assert sum(1 for c in sniffed.cells if c.cell_type == "code") == 0


def test_hidden_silencer_mutes_noisy_loggers_and_leaves_no_trace(docs_build, tmp_path):
    """The executor injects a hidden logging-silencer cell before execution
    and strips it afterwards: noisy-library INFO logging (RapidOCR-style)
    must not reach harvested outputs, and the returned notebook must contain
    exactly the page's own cells."""
    # Mimic what RapidOCR actually does: attach its own StreamHandler and
    # reset its own level at import/init time — AFTER the silencer cell ran.
    # The silencer's filter on the emitting logger must still block it.
    body = (
        "# T\n"
        "\n"
        "```python\n"
        "import logging, sys\n"
        'noisy = logging.getLogger("RapidOCR")\n'
        "noisy.setLevel(logging.INFO)\n"
        "noisy.addHandler(logging.StreamHandler(sys.stdout))\n"
        'noisy.info("model chatter")\n'
        'print("real output")\n'
        "```\n"
    )
    nodes = docs_build.parse_document(body)
    nb = docs_build.execute_page_notebook(nodes, workdir=tmp_path, timeout=120)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 1
    assert "RapidOCR" not in code_cells[0].source or "getLogger" in code_cells[0].source
    text = "".join(
        out.get("text", "") for out in code_cells[0].get("outputs", []) if isinstance(out, dict)
    )
    assert "real output" in text
    assert "model chatter" not in text
    assert all(c.get("metadata", {}).get("npdf") != "hidden-setup" for c in nb.cells)
