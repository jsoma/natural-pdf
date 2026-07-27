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

Caching
-------
``.docs-build-cache.json`` at the repo root maps each source page to the
sha256 of its markdown plus the installed natural-pdf version; when both
match and all recorded outputs still exist the page is skipped. ``--force``
bypasses the cache.

Usage
-----
    uv run python scripts/docs_build.py <src.md|dir> --out <outdir> \
        [--force] [--no-execute] [--keep-going] [--timeout N]

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
from nbclient.exceptions import CellExecutionError
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_FILE = REPO_ROOT / ".docs-build-cache.json"
DEFAULT_TIMEOUT = 600
THUMB_MAX_WIDTH = 480
COLAB_REPO = "jsoma/natural-pdf"
COLAB_BRANCH = "main"
PIP_INSTALL_CELL = '%pip install "natural-pdf[all]"'
ASSET_HASH_LENGTH = 12

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


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
    """Promote docs metadata alias ``skip: true`` to nbclient's skip tag.

    Same normalization as scripts/01-execute_notebooks.py.
    """
    for cell in notebook.cells:
        metadata = cell.setdefault("metadata", {})
        if metadata.get("skip"):
            tags = metadata.setdefault("tags", [])
            if "skip-execution" not in tags:
                tags.append("skip-execution")


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


def execute_page_notebook(
    nodes: List[Node],
    *,
    workdir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    kernel_name: str = "python3",
) -> Any:
    """Execute the page's python fences as notebook cells; return the
    executed notebook. Raises DocsBuildError on any cell error."""
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

    from jupyter_client.manager import KernelManager

    km = KernelManager(
        kernel_name=kernel_name,
        kernel_spec_manager=_make_current_env_kernel_spec_manager(),
    )
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        km=km,
        resources={"metadata": {"path": str(workdir)}},
    )
    try:
        client.execute()
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


def render_cell_outputs(cell: Any, assets_dir: Path, asset_ref_prefix: str) -> RenderedOutputs:
    """Turn a single executed code cell's outputs into markdown snippets.

    Priority per rich output: image/png > text/html > text/plain. Stream text
    and text/plain results are pooled into fenced ``output`` blocks.
    """
    rendered = RenderedOutputs()
    text_buffer: List[str] = []

    def flush_text() -> None:
        if not text_buffer:
            return
        text = strip_ansi("".join(text_buffer))
        # Keep only content after the last carriage return per line
        # (progress-bar style rewrites), then trim trailing blank space.
        lines = [seg.split("\r")[-1] for seg in text.split("\n")]
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
                rendered.snippets.append(f"![]({asset_ref_prefix}/{name})")
            elif "text/html" in data:
                flush_text()
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
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


def colab_url(notebook_path: Path) -> str:
    try:
        rel = notebook_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = f"notebooks/{notebook_path.name}"
    return f"https://colab.research.google.com/github/{COLAB_REPO}/blob/{COLAB_BRANCH}/{rel}"


def build_notebook_artifact(nodes: List[Node], notebook_path: Path) -> Any:
    """Clean (unexecuted) .ipynb: Colab badge cell + pip-install cell + page."""
    nb_md = nodes_to_notebook_markdown(nodes)
    notebook = jupytext.reads(nb_md, fmt="md")

    badge = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url(notebook_path)})"
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


def _cache_key(src: Path) -> str:
    src = src.resolve()
    try:
        return src.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(src)


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
) -> PageResult:
    """Build all artifacts for one authored markdown page."""
    start = time.monotonic()
    src = Path(src)
    page = src.stem
    out_dir = Path(out_dir)
    workdir = Path(workdir) if workdir else Path.cwd()

    text = src.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cache = load_cache(cache_file)
    key = _cache_key(src)
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

    md_out = out_dir / f"{page}.md"
    assets_dir = out_dir / "assets" / page
    notebook_out = out_dir / "notebooks" / f"{page}.ipynb"
    thumb_out = out_dir / "assets" / f"{page}-thumb.png"

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
        # Rebuild the page's assets dir from scratch so stale images vanish.
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        for node, cell in zip(iter_code_nodes(nodes), code_cells):
            rendered = render_cell_outputs(cell, assets_dir, f"assets/{page}")
            outputs_by_index[node.index] = rendered
            all_images.extend(rendered.images)

    # 1. Executed markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_executed_markdown(nodes, outputs_by_index), encoding="utf-8")
    produced: List[Path] = [md_out]
    produced.extend(sorted(assets_dir.glob("*.png")) if assets_dir.exists() else [])

    # 2. Notebook artifact
    notebook_out.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook_artifact(nodes, notebook_out)
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


def collect_sources(target: Path) -> List[Path]:
    if target.is_dir():
        return sorted(p for p in target.rglob("*.md") if p.is_file())
    return [target]


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
        "--cache-file", type=Path, default=DEFAULT_CACHE_FILE, help="Cache JSON path"
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        parser.error(f"Source not found: {args.source}")

    sources = collect_sources(args.source)
    if not sources:
        print(f"No markdown files found under {args.source}")
        return 0

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
