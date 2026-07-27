#!/usr/bin/env python
"""docs_build.py — executed-markdown pipeline for the natural-pdf docs.

Turns an authored markdown page into three artifacts:

1. Executed markdown + assets: the page with each ``python`` fence followed by
   its rendered outputs. PIL images are saved as content-addressed PNGs in
   ``<out>/assets/<page>/<sha256-prefix>.png`` and injected as image refs;
   pandas DataFrames are injected as their HTML repr (raw HTML is fine for
   mkdocs-material); stdout / plain-text results become a fenced ``output``
   block. Bare trailing expressions render (notebook semantics — each fence
   is executed as a notebook cell via jupytext + nbclient).
2. An unexecuted ``.ipynb`` (jupytext conversion of the page) with a prepended
   "Open in Colab" badge cell and a ``%pip install "natural-pdf[all]"`` cell.
3. An optional thumbnail: frontmatter ``thumbnail: N`` picks the Nth image
   output on the page (1-indexed) and writes ``<out>/assets/<page>-thumb.png``
   downscaled to at most 480px wide.

Frontmatter keys (stripped from every artifact — user-visible outputs never
contain it):

    fixture: pdfs/01-practice.pdf   # informational only
    thumbnail: 2                    # optional, 1-indexed image output
    skip: true                      # skip execution; page passes through
    tier: fast                      # CI tier: fast (default) or nightly

Fence attributes (jupytext cell-attribute syntax; stripped from the rendered
fence in the executed markdown):

    ```python skip=true         # cell is not executed (nbclient skip tag)
    ```python hide-output       # cell executes; its outputs are omitted
    ```python alt="A chart"     # alt text for images this fence produces
                                # (2nd image gets 'A chart (2)', etc.)

When ``alt`` is absent, images default their alt text to the nearest
preceding markdown heading on the page (empty if there is none).

Output hygiene: pandas ``<style scoped>`` blocks are stripped from DataFrame
HTML outputs (the table is kept), and trailing whitespace is trimmed from
each line inside ``output`` blocks (internal alignment is preserved).

Colab: in the .ipynb artifact ONLY, quoted committed-fixture paths in code
cells ("pdfs/<name>.pdf") are rewritten to
https://raw.githubusercontent.com/jsoma/natural-pdf/main/pdfs/<name>.pdf so
the notebook runs standalone; the executed markdown keeps the local path.

Tab blocks
----------
pymdown ``/// tab | Title`` ... ``///`` blocks are supported. v1 CAVEAT: the
fences inside every tab execute SEQUENTIALLY in the page's single shared
namespace, in document order — tabs are NOT isolated from each other, so a
variable assigned in tab A is visible in tab B, and two tabs that assign the
same name will interfere. Forked-namespace (per-tab) execution is vNext.
Tab markup survives into the executed markdown (outputs are injected inside
the correct tab); in the .ipynb artifact each tab becomes a markdown cell
``**Title**`` followed by its code cells.

Determinism
-----------
Every captured PNG is re-encoded through PIL with no ancillary metadata
(no tEXt/tIME chunks), so identical pixels produce identical bytes, the
content-addressed filename is stable, and re-running the build causes no
diff churn.

Output layout
-------------
Directory builds mirror the source tree under ``--out``: ``<root>/x/index.md``
lands at ``<out>/x/index.md`` with assets in ``<out>/x/assets/index/`` and the
notebook in ``<out>/x/notebooks/index.ipynb``, so same-named pages in
different subdirectories cannot collide. Single-file builds write directly
into ``--out``. A build that would overwrite the authored page is an error.

Caching
-------
``.docs-build-cache.json`` at the repo root maps each (source page, output
destination) pair to the sha256 of the page's markdown plus the installed
natural-pdf version; when both match and all recorded outputs still exist
the page is skipped. ``--force`` bypasses the cache.

Usage
-----
    uv run python scripts/docs_build.py <src.md|dir> --out <outdir> \
        [--force] [--no-execute] [--keep-going] [--timeout N] [--tier fast|nightly] \
        [--colab-ref REF] [--colab-prefix PATH]

``--out`` must not sit inside a directory source (the next run would
rediscover and reprocess its own output); discovery also always skips the
out dir itself. ``--colab-ref`` / ``--colab-prefix`` configure the Colab
badge target (see ``colab_url``).

``--tier`` builds only pages whose frontmatter ``tier`` matches; pages that
declare no tier count as ``fast``. An unknown tier value on a page is an
error (a typo would otherwise silently drop the page from CI).

Execution errors fail the build loudly (page, cell index, traceback);
``--keep-going`` collects all failures across a directory before exiting
nonzero.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nbformat
import yaml
from jupytext import jupytext
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_FILE = REPO_ROOT / ".docs-build-cache.json"
DEFAULT_TIMEOUT = 600
THUMB_MAX_WIDTH = 480
COLAB_REPO = "jsoma/natural-pdf"
COLAB_BRANCH = "main"
# Colab badge defaults: the exported notebooks are assumed to be deployed (by
# CI) at the root of the gh-pages branch, mirroring the output tree. Override
# with --colab-ref / --colab-prefix when the deployment layout differs.
COLAB_DEFAULT_REF = "gh-pages"
COLAB_DEFAULT_PREFIX = ""
PIP_INSTALL_CELL = '%pip install "natural-pdf[all]"'
SKIP_CELL_COMMENT = "# This cell is illustrative — requires client/API setup"
ASSET_HASH_LENGTH = 12
RAW_FIXTURE_BASE = f"https://raw.githubusercontent.com/{COLAB_REPO}/{COLAB_BRANCH}"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Quoted committed-fixture path in code, e.g. PDF("pdfs/01-practice.pdf").
FIXTURE_PATH_RE = re.compile(r"(['\"])pdfs/([^'\"]+?\.pdf)\1")
# Pandas DataFrame reprs carry a <style scoped> block mkdocs doesn't need.
STYLE_SCOPED_RE = re.compile(r"<style scoped>.*?</style>\s*", re.DOTALL)
# ATX heading; group(2) is the heading text (trailing #'s tolerated).
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Fence attributes.
ALT_ATTR_RE = re.compile(r'(?:^|\s)alt="([^"]*)"')
HIDE_OUTPUT_ATTR_RE = re.compile(r"(?:^|\s)hide-output(?:=true)?(?=\s|$)")
HIDE_OUTPUT_TAG = "hide-output"


class DocsBuildError(Exception):
    """A page failed to build. Message carries page + cell + traceback."""


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split optional YAML frontmatter off the top of a markdown page.

    Returns (metadata dict, body). Missing/invalid frontmatter yields ({}, text).
    """
    match = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", text, flags=re.DOTALL)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[match.end() :]


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


@dataclass
class MarkdownNode:
    text: str


@dataclass
class CodeNode:
    code: str
    info: str  # full fence info string, e.g. 'python' or 'python skip=true'
    index: int  # 0-based ordinal among python fences, document order


@dataclass
class TabNode:
    title: str
    children: List[Union[MarkdownNode, "CodeNode"]] = field(default_factory=list)


Node = Union[MarkdownNode, CodeNode, TabNode]

FENCE_OPEN_RE = re.compile(r"^```(\S.*)?$")
PYTHON_INFO_RE = re.compile(r"^python\b")
TAB_OPEN_RE = re.compile(r"^///\s*tab\s*\|\s*(.+?)\s*$")
TAB_CLOSE_RE = re.compile(r"^///\s*$")


def parse_document(body: str) -> List[Node]:
    """Parse a markdown body into markdown / python-code / tab nodes.

    Only flush-left ``` fences are recognized. Non-python fences stay part of
    the surrounding markdown. Tabs may not nest (v1).
    """
    nodes: List[Node] = []
    lines = body.split("\n")
    i = 0
    code_counter = 0
    current_tab: Optional[TabNode] = None
    md_buffer: List[str] = []

    def flush_markdown() -> None:
        if not md_buffer:
            return
        text = "\n".join(md_buffer).strip("\n")
        md_buffer.clear()
        if text.strip() == "":
            return
        target = current_tab.children if current_tab is not None else nodes
        target.append(MarkdownNode(text))

    while i < len(lines):
        line = lines[i]
        fence = FENCE_OPEN_RE.match(line)
        if fence:
            info = (fence.group(1) or "").strip()
            # Collect until closing fence
            block: List[str] = []
            i += 1
            closed = False
            while i < len(lines):
                if lines[i].rstrip() == "```":
                    closed = True
                    i += 1
                    break
                block.append(lines[i])
                i += 1
            if not closed:
                raise DocsBuildError(f"Unclosed code fence (info={info!r})")
            if PYTHON_INFO_RE.match(info):
                flush_markdown()
                node = CodeNode(code="\n".join(block), info=info, index=code_counter)
                code_counter += 1
                target = current_tab.children if current_tab is not None else nodes
                target.append(node)
            else:
                # Non-python fence: passthrough markdown, verbatim
                md_buffer.append(line)
                md_buffer.extend(block)
                md_buffer.append("```")
            continue

        tab_open = TAB_OPEN_RE.match(line)
        if tab_open:
            if current_tab is not None:
                raise DocsBuildError("Nested /// tab blocks are not supported (v1)")
            flush_markdown()
            current_tab = TabNode(title=tab_open.group(1))
            i += 1
            continue

        if current_tab is not None and TAB_CLOSE_RE.match(line):
            flush_markdown()
            nodes.append(current_tab)
            current_tab = None
            i += 1
            continue

        md_buffer.append(line)
        i += 1

    if current_tab is not None:
        raise DocsBuildError(f"Unclosed /// tab block: {current_tab.title!r}")
    flush_markdown()
    return nodes


def count_code_nodes(nodes: List[Node]) -> int:
    total = 0
    for node in nodes:
        if isinstance(node, CodeNode):
            total += 1
        elif isinstance(node, TabNode):
            total += sum(1 for c in node.children if isinstance(c, CodeNode))
    return total


def iter_code_nodes(nodes: List[Node]):
    for node in nodes:
        if isinstance(node, CodeNode):
            yield node
        elif isinstance(node, TabNode):
            for child in node.children:
                if isinstance(child, CodeNode):
                    yield child


# ---------------------------------------------------------------------------
# Markdown views of the document (for execution and the notebook artifact)
# ---------------------------------------------------------------------------


def _code_fence(node: CodeNode, include_info: bool = True) -> str:
    info = node.info if include_info else "python"
    return f"```{info}\n{node.code}\n```"


def nodes_to_execution_markdown(nodes: List[Node]) -> str:
    """Flatten the doc (tab markers dropped) into markdown for jupytext.

    The nth python fence here becomes the nth code cell of the executed
    notebook — tabs run sequentially in the shared page namespace (see module
    docstring caveat).
    """
    parts: List[str] = []
    for node in nodes:
        if isinstance(node, MarkdownNode):
            parts.append(node.text)
        elif isinstance(node, CodeNode):
            parts.append(_code_fence(node))
        else:  # TabNode
            for child in node.children:
                if isinstance(child, MarkdownNode):
                    parts.append(child.text)
                else:
                    parts.append(_code_fence(child))
    return "\n\n".join(parts) + "\n"


def nodes_to_notebook_markdown(nodes: List[Node]) -> str:
    """Markdown for the .ipynb artifact: each tab becomes a bold-title
    markdown paragraph followed by its cells."""
    parts: List[str] = []
    for node in nodes:
        if isinstance(node, MarkdownNode):
            parts.append(node.text)
        elif isinstance(node, CodeNode):
            parts.append(_code_fence(node))
        else:
            parts.append(f"**{node.title}**")
            for child in node.children:
                if isinstance(child, MarkdownNode):
                    parts.append(child.text)
                else:
                    parts.append(_code_fence(child))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Execution (jupytext -> nbclient)
# ---------------------------------------------------------------------------


def normalize_notebook_metadata(notebook: Any) -> None:
    """Promote docs fence attributes to cell tags.

    ``skip=true`` becomes nbclient's ``skip-execution`` tag (same
    normalization as scripts/01-execute_notebooks.py). ``hide-output`` (bare
    or ``=true`` — jupytext parses the bare form into
    ``incorrectly_encoded_metadata``) becomes the ``hide-output`` tag, which
    the output harvester honors by dropping the cell's outputs.
    """
    for cell in notebook.cells:
        metadata = cell.setdefault("metadata", {})
        if metadata.get("skip"):
            tags = metadata.setdefault("tags", [])
            if "skip-execution" not in tags:
                tags.append("skip-execution")
        hide = bool(metadata.get("hide-output") or metadata.get("hide_output"))
        if not hide:
            stray = metadata.get("incorrectly_encoded_metadata")
            if isinstance(stray, str) and HIDE_OUTPUT_ATTR_RE.search(stray):
                hide = True
        if hide:
            tags = metadata.setdefault("tags", [])
            if HIDE_OUTPUT_TAG not in tags:
                tags.append(HIDE_OUTPUT_TAG)


def cell_hides_output(cell: Any) -> bool:
    return HIDE_OUTPUT_TAG in cell.get("metadata", {}).get("tags", [])


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_current_env_kernel_spec_manager():
    """KernelSpecManager subclass that pins ``python3`` to the interpreter
    running this build (sys.executable), so a user-level ``python3``
    kernelspec pointing at some other environment can't hijack docs
    execution. Other kernel names resolve normally."""
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    class CurrentEnvKernelSpecManager(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            if kernel_name == "python3":
                from ipykernel.kernelspec import RESOURCES, get_kernel_dict

                return KernelSpec(resource_dir=RESOURCES, **get_kernel_dict())
            return super().get_kernel_spec(kernel_name)

    return CurrentEnvKernelSpecManager()


class _DocsNotebookClient(NotebookClient):
    """NotebookClient that pins the python3 kernelspec to sys.executable.

    The client creates and OWNS its kernel manager (nbclient's default
    AsyncKernelManager), so ``execute()`` starts the kernel, enforces
    per-cell ``timeout``, and shuts the kernel down when it finishes —
    passing an external KernelManager disables all of that (``owns_km`` is
    False, so setup_kernel never cleans up: one leaked kernel per page and a
    stray '[IPKernelApp] WARNING | Parent appears to have exited' at exit).
    """

    def create_kernel_manager(self):
        km = super().create_kernel_manager()
        km.kernel_spec_manager = _make_current_env_kernel_spec_manager()
        return km


def _shutdown_kernel(client: NotebookClient) -> None:
    """Best-effort kernel teardown (no-op when execute() already cleaned up)."""
    try:
        km = getattr(client, "km", None)
        if km is not None and km.has_kernel:
            client._cleanup_kernel()
    except Exception:
        pass


def execute_page_notebook(
    nodes: List[Node],
    *,
    workdir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    kernel_name: str = "python3",
) -> Any:
    """Execute the page's python fences as notebook cells; return the
    executed notebook. Raises DocsBuildError on any cell error or timeout.
    The kernel is always shut down before this function returns."""
    exec_md = nodes_to_execution_markdown(nodes)
    notebook = jupytext.reads(exec_md, fmt="md")
    normalize_notebook_metadata(notebook)

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    expected = count_code_nodes(nodes)
    if len(code_cells) != expected:
        raise DocsBuildError(
            f"Internal fence/cell mismatch: parsed {expected} python fences "
            f"but jupytext produced {len(code_cells)} code cells"
        )

    client = _DocsNotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(workdir)}},
    )
    try:
        client.execute()
    except CellTimeoutError as exc:
        raise DocsBuildError(
            f"Execution timed out (timeout={timeout}s per cell):\n{strip_ansi(str(exc))}"
        ) from exc
    except CellExecutionError as exc:
        # Locate the failing code cell for a precise report.
        failing_ordinal = None
        traceback_text = strip_ansi(str(exc))
        for ordinal, cell in enumerate(code_cells):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    failing_ordinal = ordinal
                    traceback_text = strip_ansi("\n".join(output.get("traceback", [])))
                    break
            if failing_ordinal is not None:
                break
        where = f"python fence #{failing_ordinal + 1}" if failing_ordinal is not None else "a cell"
        snippet = ""
        if failing_ordinal is not None:
            first_line = code_cells[failing_ordinal].source.splitlines()[:1]
            snippet = f" ({first_line[0]!r} ...)" if first_line else ""
        raise DocsBuildError(f"Execution failed in {where}{snippet}:\n{traceback_text}") from exc
    finally:
        _shutdown_kernel(client)
    return notebook


# ---------------------------------------------------------------------------
# Output harvesting / rendering
# ---------------------------------------------------------------------------


def normalize_png(data: bytes) -> bytes:
    """Re-encode PNG bytes through PIL without ancillary metadata so identical
    pixels always produce identical bytes."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return buf.getvalue()


def asset_filename(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()[:ASSET_HASH_LENGTH] + ".png"


def save_png_asset(png_data: bytes, assets_dir: Path) -> str:
    """Save (deterministically re-encoded) PNG content-addressed; return filename."""
    normalized = normalize_png(png_data)
    name = asset_filename(normalized)
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / name
    if not path.exists():
        path.write_bytes(normalized)
    return name


@dataclass
class RenderedOutputs:
    snippets: List[str] = field(default_factory=list)  # markdown snippets, in order
    images: List[str] = field(default_factory=list)  # asset filenames, in order


def compute_default_alts(nodes: List[Node]) -> Dict[int, str]:
    """Map each python fence index to the nearest preceding markdown heading.

    Used as the default alt text for images a fence produces when the fence
    has no ``alt="..."`` attribute. Headings inside passthrough (non-python)
    fences are ignored. Fences before the first heading map to "".
    """
    alts: Dict[int, str] = {}
    last_heading = ""

    def scan_markdown(text: str) -> None:
        nonlocal last_heading
        in_fence = False
        for line in text.split("\n"):
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = HEADING_RE.match(line)
            if match:
                last_heading = match.group(2).strip()

    def visit(children: List[Node]) -> None:
        for node in children:
            if isinstance(node, MarkdownNode):
                scan_markdown(node.text)
            elif isinstance(node, CodeNode):
                alts[node.index] = last_heading
            else:
                visit(node.children)

    visit(nodes)
    return alts


def fence_alt(node: CodeNode, default_alts: Dict[int, str]) -> str:
    """Alt text for a fence's images: explicit ``alt="..."`` attribute wins,
    else the nearest preceding heading, else ""."""
    match = ALT_ATTR_RE.search(node.info)
    if match:
        return match.group(1)
    return default_alts.get(node.index, "")


def render_cell_outputs(
    cell: Any, assets_dir: Path, asset_ref_prefix: str, alt: str = ""
) -> RenderedOutputs:
    """Turn a single executed code cell's outputs into markdown snippets.

    Priority per rich output: image/png > text/html > text/plain. Stream text
    and text/plain results are pooled into fenced ``output`` blocks with
    line-trailing whitespace trimmed (internal alignment preserved). Pandas
    ``<style scoped>`` blocks are stripped from HTML outputs. ``alt`` is
    applied to the fence's images verbatim; a second image gets "alt (2)",
    a third "alt (3)", and so on.
    """
    rendered = RenderedOutputs()
    text_buffer: List[str] = []

    def flush_text() -> None:
        if not text_buffer:
            return
        text = strip_ansi("".join(text_buffer))
        # Keep only content after the last carriage return per line
        # (progress-bar style rewrites), trim each line's trailing whitespace
        # (avoids horizontal scroll from padded columns), then trim trailing
        # blank space.
        lines = [seg.split("\r")[-1].rstrip() for seg in text.split("\n")]
        text = "\n".join(lines).rstrip()
        text_buffer.clear()
        if text:
            rendered.snippets.append(f"```output\n{text}\n```")

    for output in cell.get("outputs", []):
        otype = output.get("output_type")
        if otype == "stream":
            text_buffer.append(output.get("text", ""))
        elif otype in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "image/png" in data:
                flush_text()
                png = base64.b64decode(data["image/png"])
                name = save_png_asset(png, assets_dir)
                rendered.images.append(name)
                ordinal = len(rendered.images)
                alt_text = alt if (ordinal == 1 or not alt) else f"{alt} ({ordinal})"
                rendered.snippets.append(f"![{alt_text}]({asset_ref_prefix}/{name})")
            elif "text/html" in data:
                flush_text()
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
                html = STYLE_SCOPED_RE.sub("", html)
                rendered.snippets.append(html.strip())
            elif "text/plain" in data:
                plain = data["text/plain"]
                if isinstance(plain, list):
                    plain = "".join(plain)
                text_buffer.append(plain if plain.endswith("\n") else plain + "\n")
    flush_text()
    return rendered


def render_executed_markdown(
    nodes: List[Node], outputs_by_index: Dict[int, RenderedOutputs]
) -> str:
    """Emit the executed page: authored markdown, fences, and injected outputs
    (inside the correct tab, when the fence lives in one)."""

    def emit_code(node: CodeNode) -> List[str]:
        parts = [_code_fence(node, include_info=False)]
        rendered = outputs_by_index.get(node.index)
        if rendered:
            parts.extend(rendered.snippets)
        return parts

    parts: List[str] = []
    for node in nodes:
        if isinstance(node, MarkdownNode):
            parts.append(node.text)
        elif isinstance(node, CodeNode):
            parts.extend(emit_code(node))
        else:
            tab_parts: List[str] = []
            for child in node.children:
                if isinstance(child, MarkdownNode):
                    tab_parts.append(child.text)
                else:
                    tab_parts.extend(emit_code(child))
            body = "\n\n".join(tab_parts)
            parts.append(f"/// tab | {node.title}\n\n{body}\n\n///")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Notebook artifact
# ---------------------------------------------------------------------------


def colab_url(
    notebook_path: Path,
    *,
    ref: str = COLAB_DEFAULT_REF,
    prefix: str = COLAB_DEFAULT_PREFIX,
    out_root: Optional[Path] = None,
) -> str:
    """Compose the Colab badge target for an exported notebook.

    URL shape:
        https://colab.research.google.com/github/<repo>/blob/<ref>/<prefix>/<rel>

    where ``<rel>`` is the notebook path relative to the output tree
    (``out_root``, i.e. ``--out``) — e.g. ``learn/notebooks/page.ipynb`` for a
    directory build, ``notebooks/page.ipynb`` for a single-file build.

    ASSUMPTION baked into the defaults: CI deploys the output tree at the
    ROOT of the ``gh-pages`` branch (ref ``gh-pages``, empty prefix), so
    ``<out>/learn/notebooks/x.ipynb`` is reachable at
    ``blob/gh-pages/learn/notebooks/x.ipynb``. When the deployed layout
    differs, CI passes the real values via ``--colab-ref``/``--colab-prefix``;
    this script only composes the URL.
    """
    rel: Optional[str] = None
    if out_root is not None:
        try:
            rel = notebook_path.resolve().relative_to(Path(out_root).resolve()).as_posix()
        except ValueError:
            rel = None
    if rel is None:
        rel = f"notebooks/{notebook_path.name}"
    path = "/".join(part.strip("/") for part in (prefix, rel) if part and part.strip("/"))
    return f"https://colab.research.google.com/github/{COLAB_REPO}/blob/{ref}/{path}"


def rewrite_fixture_paths_for_colab(source: str) -> str:
    """Rewrite quoted committed-fixture paths ("pdfs/<name>.pdf") in code to
    raw GitHub URLs so the notebook runs standalone on Colab. Fixtures that
    are already URLs don't match and pass through untouched."""
    return FIXTURE_PATH_RE.sub(
        lambda m: f"{m.group(1)}{RAW_FIXTURE_BASE}/pdfs/{m.group(2)}{m.group(1)}",
        source,
    )


def apply_skip_execution_tags(notebook: Any) -> None:
    """Convert ``skip=true`` fence metadata into Jupyter's ``skip-execution``
    cell tag in the EXPORTED notebook, so "Run All" (in frontends/executors
    that honor the tag, e.g. nbclient) does not run cells the docs build never
    executed. When the cell does not already open with a comment, a one-line
    note is prepended so readers see why the cell is inert; cells that start
    with their own comment just get the tag."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        metadata = cell.get("metadata", {})
        if not metadata.get("skip"):
            continue
        tags = metadata.setdefault("tags", [])
        if "skip-execution" not in tags:
            tags.append("skip-execution")
        metadata.pop("skip", None)
        first_line = cell.source.lstrip().splitlines()[:1]
        if not (first_line and first_line[0].startswith("#")):
            cell.source = f"{SKIP_CELL_COMMENT}\n{cell.source}"


def build_notebook_artifact(
    nodes: List[Node],
    notebook_path: Path,
    *,
    colab_ref: str = COLAB_DEFAULT_REF,
    colab_prefix: str = COLAB_DEFAULT_PREFIX,
    out_root: Optional[Path] = None,
) -> Any:
    """Clean (unexecuted) .ipynb: Colab badge cell + pip-install cell + page.

    Code cells get committed-fixture paths rewritten to raw GitHub URLs
    (.ipynb artifact only — the executed markdown keeps local paths), and
    ``skip=true`` fences become ``skip-execution``-tagged cells. The badge
    URL is composed by :func:`colab_url` (see its docstring for the deployed
    layout assumption behind the defaults)."""
    nb_md = nodes_to_notebook_markdown(nodes)
    notebook = jupytext.reads(nb_md, fmt="md")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = rewrite_fixture_paths_for_colab(cell.source)
    apply_skip_execution_tags(notebook)

    badge = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url(notebook_path, ref=colab_ref, prefix=colab_prefix, out_root=out_root)})"
    )
    prepend = [
        nbformat.v4.new_markdown_cell(badge),
        nbformat.v4.new_code_cell(PIP_INSTALL_CELL),
    ]
    notebook.cells = prepend + list(notebook.cells)
    notebook.metadata.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    notebook.metadata.setdefault("language_info", {"name": "python"})
    # jupytext stores its own metadata; drop it so the artifact is clean.
    notebook.metadata.pop("jupytext", None)
    # Deterministic cell ids (nbformat randomizes them -> diff churn).
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{i}"
    return notebook


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------


def write_thumbnail(image_path: Path, thumb_path: Path) -> None:
    with Image.open(image_path) as img:
        img.load()
        if img.width > THUMB_MAX_WIDTH:
            new_height = max(1, round(img.height * THUMB_MAX_WIDTH / img.width))
            img = img.resize((THUMB_MAX_WIDTH, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def natural_pdf_version() -> str:
    try:
        from importlib.metadata import version

        return version("natural-pdf")
    except Exception:
        return "unknown"


def load_cache(cache_file: Path) -> Dict[str, Any]:
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_cache(cache_file: Path, cache: Dict[str, Any]) -> None:
    cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_token(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _cache_key(src: Path, page_out_dir: Path) -> str:
    """Cache identity is (source page, output destination): building the same
    page into a second --out must not report 'cached' without writing."""
    return f"{_path_token(src)} -> {_path_token(page_out_dir)}"


def cache_entry_valid(entry: Optional[Dict[str, Any]], source_hash: str) -> bool:
    if not entry:
        return False
    if entry.get("hash") != source_hash:
        return False
    if entry.get("natural_pdf_version") != natural_pdf_version():
        return False
    outputs = entry.get("outputs", [])
    if not outputs:
        return False
    return all(Path(p).exists() for p in outputs)


# ---------------------------------------------------------------------------
# Page build orchestration
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    src: Path
    status: str  # "built", "cached", "skipped" (frontmatter skip), "failed"
    elapsed: float = 0.0
    outputs: List[Path] = field(default_factory=list)
    error: Optional[str] = None


def build_page(
    src: Path,
    out_dir: Path,
    *,
    force: bool = False,
    execute: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    kernel_name: str = "python3",
    workdir: Optional[Path] = None,
    cache_file: Path = DEFAULT_CACHE_FILE,
    source_root: Optional[Path] = None,
    colab_ref: str = COLAB_DEFAULT_REF,
    colab_prefix: str = COLAB_DEFAULT_PREFIX,
) -> PageResult:
    """Build all artifacts for one authored markdown page.

    ``source_root`` (directory builds) mirrors the source tree under
    ``out_dir``: ``<root>/solve/index.md`` lands at ``<out>/solve/index.md``
    with assets in ``<out>/solve/assets/index/`` — so two ``index.md`` pages
    in different subdirectories cannot overwrite each other, and the
    asset/notebook links inside the executed markdown stay relative to the
    page as before. Single-file builds (source_root=None) write directly
    into ``out_dir``, unchanged from v1.
    """
    start = time.monotonic()
    src = Path(src)
    page = src.stem
    out_dir = Path(out_dir)
    workdir = Path(workdir) if workdir else Path.cwd()

    if source_root is not None:
        try:
            rel_parent = src.resolve().parent.relative_to(Path(source_root).resolve())
        except ValueError:
            raise DocsBuildError(f"{src}: page is not under source root {source_root}")
        page_out_dir = out_dir / rel_parent
    else:
        page_out_dir = out_dir

    md_out = page_out_dir / f"{page}.md"
    if md_out.resolve() == src.resolve():
        raise DocsBuildError(
            f"{src}: output would overwrite the authored page — choose a different --out"
        )

    text = src.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cache = load_cache(cache_file)
    key = _cache_key(src, page_out_dir)
    if execute and not force and cache_entry_valid(cache.get(key), source_hash):
        entry = cache[key]
        return PageResult(
            src=src,
            status="cached",
            elapsed=time.monotonic() - start,
            outputs=[Path(p) for p in entry["outputs"]],
        )

    meta, body = parse_frontmatter(text)
    nodes = parse_document(body)

    should_execute = execute and not bool(meta.get("skip"))

    assets_dir = page_out_dir / "assets" / page
    notebook_out = page_out_dir / "notebooks" / f"{page}.ipynb"
    thumb_out = page_out_dir / "assets" / f"{page}-thumb.png"

    outputs_by_index: Dict[int, RenderedOutputs] = {}
    all_images: List[str] = []  # asset filenames in document order

    if should_execute:
        try:
            executed = execute_page_notebook(
                nodes, workdir=workdir, timeout=timeout, kernel_name=kernel_name
            )
        except DocsBuildError as exc:
            raise DocsBuildError(f"{src}: {exc}") from exc
        code_cells = [c for c in executed.cells if c.cell_type == "code"]
        default_alts = compute_default_alts(nodes)
        # Rebuild the page's assets dir from scratch so stale images vanish.
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        for node, cell in zip(iter_code_nodes(nodes), code_cells):
            if cell_hides_output(cell):
                # hide-output: the fence ran, but nothing is injected.
                outputs_by_index[node.index] = RenderedOutputs()
                continue
            rendered = render_cell_outputs(
                cell, assets_dir, f"assets/{page}", alt=fence_alt(node, default_alts)
            )
            outputs_by_index[node.index] = rendered
            all_images.extend(rendered.images)

    # 1. Executed markdown
    page_out_dir.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_executed_markdown(nodes, outputs_by_index), encoding="utf-8")
    produced: List[Path] = [md_out]
    produced.extend(sorted(assets_dir.glob("*.png")) if assets_dir.exists() else [])

    # 2. Notebook artifact
    notebook_out.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook_artifact(
        nodes,
        notebook_out,
        colab_ref=colab_ref,
        colab_prefix=colab_prefix,
        out_root=out_dir,
    )
    with open(notebook_out, "w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)
    produced.append(notebook_out)

    # 3. Thumbnail
    thumbnail_n = meta.get("thumbnail")
    if thumbnail_n is not None and should_execute:
        try:
            thumbnail_n = int(thumbnail_n)
        except (TypeError, ValueError):
            raise DocsBuildError(f"{src}: frontmatter 'thumbnail' must be an integer")
        if not 1 <= thumbnail_n <= len(all_images):
            raise DocsBuildError(
                f"{src}: thumbnail: {thumbnail_n} but page produced "
                f"{len(all_images)} image output(s)"
            )
        write_thumbnail(assets_dir / all_images[thumbnail_n - 1], thumb_out)
        produced.append(thumb_out)
    elif thumbnail_n is None and thumb_out.exists():
        # The page no longer declares a thumbnail: drop the stale artifact a
        # previous build left behind (creation-only handling would leak it).
        thumb_out.unlink()

    status = "built" if should_execute else "skipped"

    if should_execute:
        cache = load_cache(cache_file)  # re-read: another page may have written
        cache[key] = {
            "hash": source_hash,
            "natural_pdf_version": natural_pdf_version(),
            "outputs": [str(p) for p in produced],
        }
        save_cache(cache_file, cache)

    return PageResult(src=src, status=status, elapsed=time.monotonic() - start, outputs=produced)


def collect_sources(target: Path, exclude_dir: Optional[Path] = None) -> List[Path]:
    """Discover authored pages. ``exclude_dir`` (the resolved --out) is never
    descended into, so a build can't rediscover its own generated output."""
    if not target.is_dir():
        return [target]
    exclude = Path(exclude_dir).resolve() if exclude_dir is not None else None
    sources: List[Path] = []
    for path in sorted(target.rglob("*.md")):
        if not path.is_file():
            continue
        if exclude is not None:
            resolved = path.resolve()
            if resolved == exclude or exclude in resolved.parents:
                continue
        sources.append(path)
    return sources


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------

DEFAULT_TIER = "fast"
KNOWN_TIERS = ("fast", "nightly")


def page_tier(src: Path) -> str:
    """Read a page's frontmatter ``tier`` (default: 'fast').

    Raises DocsBuildError on values outside KNOWN_TIERS so a typo can't
    silently drop a page from every CI tier.
    """
    meta, _ = parse_frontmatter(src.read_text(encoding="utf-8"))
    tier = str(meta.get("tier", DEFAULT_TIER)).strip().lower()
    if tier not in KNOWN_TIERS:
        raise DocsBuildError(
            f"{src}: frontmatter 'tier' must be one of {', '.join(KNOWN_TIERS)} (got {tier!r})"
        )
    return tier


def filter_sources_by_tier(sources: List[Path], tier: str) -> Tuple[List[Path], List[Path]]:
    """Split sources into (matching tier, skipped)."""
    selected: List[Path] = []
    skipped: List[Path] = []
    for src in sources:
        (selected if page_tier(src) == tier else skipped).append(src)
    return selected, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build executed docs pages (markdown + assets, notebook, thumbnail)."
    )
    parser.add_argument("source", type=Path, help="Authored .md file or a directory of them")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Ignore the cache")
    parser.add_argument(
        "--no-execute", action="store_true", help="Skip execution; pass pages through"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Collect all page failures before exiting nonzero",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument(
        "--tier",
        choices=KNOWN_TIERS,
        help=(
            "Only build pages whose frontmatter 'tier' matches "
            f"(pages without a tier count as {DEFAULT_TIER!r})"
        ),
    )
    parser.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE_FILE, help="Cache JSON path"
    )
    parser.add_argument(
        "--colab-ref",
        default=COLAB_DEFAULT_REF,
        help=(
            "Git ref the Colab badge links into "
            f"(default: {COLAB_DEFAULT_REF!r}; see colab_url docstring)"
        ),
    )
    parser.add_argument(
        "--colab-prefix",
        default=COLAB_DEFAULT_PREFIX,
        help=(
            "Path prefix inside the ref where the output tree is deployed "
            "(default: the output tree sits at the ref root)"
        ),
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        parser.error(f"Source not found: {args.source}")

    if args.source.is_dir():
        src_resolved = args.source.resolve()
        out_resolved = args.out.resolve()
        if out_resolved == src_resolved or src_resolved in out_resolved.parents:
            parser.error(
                f"--out {args.out} is inside the source directory {args.source}: "
                "later runs would rediscover and reprocess the generated output "
                "(generated/generated/...). Choose an --out outside the source tree."
            )

    # Belt-and-braces for equal/edge roots: never discover our own output.
    sources = collect_sources(args.source, exclude_dir=args.out)
    if not sources:
        print(f"No markdown files found under {args.source}")
        return 0

    if args.tier:
        try:
            sources, tier_skipped = filter_sources_by_tier(sources, args.tier)
        except DocsBuildError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for src in tier_skipped:
            print(f"[tier   ] {src} (not tier={args.tier})")
        if not sources:
            print(f"No pages with tier={args.tier!r} under {args.source}")
            return 0

    # Directory builds mirror the source tree under --out (see build_page).
    source_root = args.source if args.source.is_dir() else None

    failures: List[Tuple[Path, str]] = []
    for src in sources:
        try:
            result = build_page(
                src,
                args.out,
                force=args.force,
                execute=not args.no_execute,
                timeout=args.timeout,
                kernel_name=args.kernel,
                cache_file=args.cache_file,
                source_root=source_root,
                colab_ref=args.colab_ref,
                colab_prefix=args.colab_prefix,
            )
        except DocsBuildError as exc:
            failures.append((src, str(exc)))
            print(f"[failed ] {exc}", file=sys.stderr)
            if not args.keep_going:
                return 1
            continue
        print(f"[{result.status:<7}] {src} ({result.elapsed:.2f}s, {len(result.outputs)} outputs)")

    if failures:
        print(f"\n{len(failures)} page(s) failed:", file=sys.stderr)
        for src, err in failures:
            print(f"  - {err.splitlines()[0] if err else f'{src}: unknown error'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
