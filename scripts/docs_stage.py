#!/usr/bin/env python3
"""Stage authored + executed docs into an Astro Starlight site tree.

Usage:
    uv run python scripts/docs_stage.py [--authored docs] [--executed docs-executed]
                                        [--site docs-site] [--allow-missing-api]

Populates <site>/src/content/docs/ (compiled pages) and <site>/public/
(static files), deleting and recreating ONLY those two directories.

Transformations per page (see docs/specs/starlight_migration_spec.md §5):
  - overlay executed body over authored page (metadata from authored file)
  - lift title into frontmatter, remove the source H1
  - drop executor-only frontmatter keys (fixture/tier/thumbnail/skip)
  - convert pymdown ``/// tab | Label`` groups to npdf-tabs comment markers
  - replace ```output fences with shaded <pre class="npdf-output"> blocks
  - convert leftover ``!!! type`` admonitions to Starlight asides
  - rewrite internal .md links to route-relative directory form
  - copy referenced images into public/ and rewrite to /natural-pdf/ URLs
  - substitute <!-- npdf-api:include id=X --> markers from temp/api-fragments/
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "/natural-pdf"
EXECUTOR_KEYS = ("fixture", "tier", "thumbnail", "skip")

ADMONITION_TYPE_MAP = {
    "note": "note",
    "info": "note",
    "abstract": "note",
    "tip": "tip",
    "hint": "tip",
    "success": "tip",
    "warning": "caution",
    "caution": "caution",
    "attention": "caution",
    "danger": "danger",
    "error": "danger",
    "bug": "danger",
}

FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
TAB_OPEN_RE = re.compile(r"^///\s*tab\s*\|\s*(.*?)\s*$")
TAB_CLOSE_RE = re.compile(r"^///\s*$")
ADMONITION_RE = re.compile(r'^!!!\s+([A-Za-z][\w-]*)(?:\s+"([^"]*)")?\s*$')
API_INCLUDE_RE = re.compile(r"^\s*<!--\s*npdf-api:include\s+id=([\w.-]+)\s*-->\s*$")
H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
OUTPUT_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*output(\s.*)?$")

MD_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(\s*([^)\s]+)((?:\s+"[^"]*")?)\s*\)')
MD_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\(\s*([^)\s]+?)((?:\s+"[^"]*")?)\s*\)')
HTML_SRC_RE = re.compile(r'(src\s*=\s*)(["\'])([^"\']+)\2')
HTML_HREF_RE = re.compile(r'(href\s*=\s*)(["\'])([^"\']+)\2')
CODE_SPAN_RE = re.compile(r"(`+)[^`]*\1")

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "//", "data:", "ftp:")


class FenceTracker:
    """Track fenced-code state line by line (``` and ~~~ fences)."""

    def __init__(self) -> None:
        self._char: str | None = None
        self._len = 0

    @property
    def in_fence(self) -> bool:
        return self._char is not None

    def feed(self, line: str) -> bool:
        """Consume a line; return True if the line is *inside* a fence
        (fence delimiter lines themselves count as inside/boundary, i.e.
        not eligible for prose transformations)."""
        m = FENCE_OPEN_RE.match(line)
        if self._char is None:
            if m and (m.group(2)[0] != "`" or "`" not in m.group(3)):
                self._char = m.group(2)[0]
                self._len = len(m.group(2))
                return True
            return False
        # inside a fence: check for close
        if m and m.group(2)[0] == self._char and len(m.group(2)) >= self._len:
            if m.group(3).strip() == "":
                self._char = None
                self._len = 0
        return True


@dataclass
class StageContext:
    authored: Path
    executed: Path
    content_dir: Path
    public_dir: Path
    fragments_dir: Path
    allow_missing_api: bool
    pages: set[str] = field(default_factory=set)  # relative page paths, posix
    errors: list[str] = field(default_factory=list)
    copied_public: set[str] = field(default_factory=set)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


# ---------------------------------------------------------------------------
# frontmatter


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). Empty dict when no frontmatter block."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            raw = "".join(lines[1:i])
            try:
                meta = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                return {}, text
            if not isinstance(meta, dict):
                meta = {}
            return meta, "".join(lines[i + 1 :])
    return {}, text


# ---------------------------------------------------------------------------
# per-line transformations


def substitute_api_includes(lines: list[str], ctx: StageContext, page: str) -> list[str]:
    out: list[str] = []
    tracker = FenceTracker()
    for line in lines:
        in_fence = tracker.feed(line)
        m = None if in_fence else API_INCLUDE_RE.match(line)
        if not m:
            out.append(line)
            continue
        frag = ctx.fragments_dir / f"{m.group(1)}.md"
        if frag.is_file():
            out.extend(frag.read_text(encoding="utf-8").splitlines())
        elif ctx.allow_missing_api:
            out.append(line)  # comment renders as nothing
        else:
            ctx.error(f"{page}: missing API fragment {frag}")
            out.append(line)
    return out


def lift_title(
    meta: dict, lines: list[str], ctx: StageContext, page: str
) -> tuple[str | None, list[str]]:
    """Determine the page title and remove the first H1 from the body."""
    explicit = meta.get("title")
    h1_index = None
    h1_text = None
    tracker = FenceTracker()
    for i, line in enumerate(lines):
        if tracker.feed(line):
            continue
        m = H1_RE.match(line)
        if m:
            h1_index = i
            h1_text = m.group(1).strip()
            break

    if explicit is not None:
        title = str(explicit)
        if h1_text is not None:
            if h1_text != title:
                ctx.error(f"{page}: frontmatter title {title!r} conflicts with H1 {h1_text!r}")
                return None, lines
            lines = _remove_heading(lines, h1_index)
        return title, lines

    if h1_text is None:
        ctx.error(f"{page}: no title (no frontmatter title and no H1 found)")
        return None, lines
    return h1_text, _remove_heading(lines, h1_index)


def _remove_heading(lines: list[str], index: int) -> list[str]:
    out = lines[:index] + lines[index + 1 :]
    # drop a now-leading blank line left behind by the heading
    while index < len(out) and index == 0 and out and out[0].strip() == "":
        out.pop(0)
    return out


def convert_admonitions(lines: list[str], ctx: StageContext, page: str) -> list[str]:
    """Convert leftover `!!! type "Title"` admonitions to Starlight asides."""
    out: list[str] = []
    tracker = FenceTracker()
    i = 0
    while i < len(lines):
        line = lines[i]
        in_fence = tracker.feed(line)
        m = None if in_fence else ADMONITION_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        kind = ADMONITION_TYPE_MAP.get(m.group(1).lower(), "note")
        title = m.group(2)
        body: list[str] = []
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() == "":
                body.append("")
                i += 1
            elif nxt.startswith("    ") or nxt.startswith("\t"):
                body.append(nxt[4:] if nxt.startswith("    ") else nxt[1:])
                i += 1
            else:
                break
        while body and body[0] == "":
            body.pop(0)
        while body and body[-1] == "":
            body.pop()
        header = f":::{kind}[{title}]" if title else f":::{kind}"
        out.append(header)
        out.extend(body)
        out.append(":::")
    return out


def convert_tabs(lines: list[str], ctx: StageContext, page: str) -> list[str]:
    """Convert consecutive `/// tab | Label` blocks into npdf-tabs markers."""
    out: list[str] = []
    tracker = FenceTracker()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        in_fence = tracker.feed(line)
        m = None if in_fence else TAB_OPEN_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        # collect a group of consecutive tab blocks
        group: list[tuple[str, list[str]]] = []
        while True:
            label = m.group(1)
            body: list[str] = []
            i += 1
            closed = False
            while i < n:
                inner = lines[i]
                inner_in_fence = tracker.feed(inner)
                if not inner_in_fence and TAB_CLOSE_RE.match(inner):
                    closed = True
                    i += 1
                    break
                body.append(inner)
                i += 1
            if not closed:
                ctx.error(f"{page}: unterminated /// tab block (label {label!r})")
            group.append((label, body))
            # look ahead past blank lines for another tab in the same group
            j = i
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n:
                m2 = TAB_OPEN_RE.match(lines[j])
                if m2:
                    for k in range(i, j):
                        tracker.feed(lines[k])
                    tracker.feed(lines[j])
                    i = j
                    m = m2
                    continue
            break

        out.append("<!-- npdf-tabs:start -->")
        for label, body in group:
            safe_label = label.replace('"', "&quot;")
            out.append(f'<!-- npdf-tab:start label="{safe_label}" -->')
            out.extend(body)
            out.append("<!-- npdf-tab:end -->")
        out.append("<!-- npdf-tabs:end -->")
    return out


def rewrite_output_fences(lines: list[str]) -> list[str]:
    """Replace ```output fenced blocks with shaded <pre class="npdf-output"> HTML.

    Executed-cell output should not render like a syntax-highlighted code
    frame; an escaped <pre> lets CSS give it a muted inline-code-style
    background instead. MUST run after link/image rewriting: the emitted raw
    HTML would otherwise be scanned by the href/src attribute rewriters
    (while fenced, the content is protected by the fence trackers).
    """
    out: list[str] = []
    tracker = FenceTracker()
    body: list[str] = []
    indent = ""
    collecting = False
    for line in lines:
        was_in_fence = tracker.in_fence
        tracker.feed(line)
        if collecting:
            if tracker.in_fence:
                body.append(line)
            else:  # this line closed the output fence
                content = html.escape("\n".join(body), quote=False)
                out.append(f'{indent}<pre class="npdf-output"><code>{content}</code></pre>')
                collecting = False
                body = []
            continue
        if not was_in_fence and tracker.in_fence:
            m = OUTPUT_FENCE_RE.match(line)
            if m:
                indent = m.group(1)
                collecting = True
                continue
        out.append(line)
    if collecting:  # unterminated fence at EOF
        content = html.escape("\n".join(body), quote=False)
        out.append(f'{indent}<pre class="npdf-output"><code>{content}</code></pre>')
    return out


# ---------------------------------------------------------------------------
# links and images


def _route_dir(page_rel: str) -> str:
    """Route directory for a page: a/b.md -> a/b, a/index.md -> a, index.md -> ''."""
    no_ext = page_rel[:-3] if page_rel.endswith(".md") else page_rel
    if posixpath.basename(no_ext) == "index":
        return posixpath.dirname(no_ext)
    return no_ext


def rewrite_md_target(target: str, page_rel: str, ctx: StageContext) -> str | None:
    """Rewrite an internal .md link target to route-relative form, or None."""
    t = target.strip()
    if not t or t.startswith(EXTERNAL_PREFIXES) or t.startswith(("#", "/")):
        return None
    path, hash_sep, frag = t.partition("#")
    if not path.endswith(".md"):
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page_rel), path))
    if resolved.startswith(".."):
        ctx.error(f"{page_rel}: link escapes docs root: {target}")
        return None
    if resolved not in ctx.pages:
        ctx.error(f"{page_rel}: link target not found: {target}")
        return None
    src_route = _route_dir(page_rel) or "."
    dst_route = _route_dir(resolved) or "."
    rel = posixpath.relpath(dst_route, src_route)
    new = "./" if rel == "." else rel + "/"
    if hash_sep:
        new += "#" + frag
    return new


def rewrite_image_src(src: str, page_rel: str, ctx: StageContext) -> str | None:
    """Copy a relative image into public/ and return its base-prefixed URL."""
    t = src.strip()
    if not t or t.startswith(EXTERNAL_PREFIXES) or t.startswith(("#", "/")):
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page_rel), t))
    if resolved.startswith(".."):
        ctx.error(f"{page_rel}: image reference escapes docs root: {src}")
        return None
    for base in (ctx.authored, ctx.executed):
        candidate = base / resolved
        if candidate.is_file():
            dest = ctx.public_dir / resolved
            if resolved not in ctx.copied_public:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, dest)
                ctx.copied_public.add(resolved)
            return f"{BASE_URL}/{resolved}"
    ctx.error(f"{page_rel}: referenced image not found: {src}")
    return None


def _transform_prose_line(line: str, page_rel: str, ctx: StageContext) -> str:
    def image_sub(m: re.Match) -> str:
        new = rewrite_image_src(m.group(2), page_rel, ctx)
        if new is None:
            return m.group(0)
        return f"![{m.group(1)}]({new}{m.group(3)})"

    def link_sub(m: re.Match) -> str:
        new = rewrite_md_target(m.group(2), page_rel, ctx)
        if new is None:
            return m.group(0)
        return f"[{m.group(1)}]({new}{m.group(3)})"

    def html_src_sub(m: re.Match) -> str:
        new = rewrite_image_src(m.group(3), page_rel, ctx)
        if new is None:
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{new}{m.group(2)}"

    def html_href_sub(m: re.Match) -> str:
        new = rewrite_md_target(m.group(3), page_rel, ctx)
        if new is None:
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{new}{m.group(2)}"

    # Apply each rewrite across the whole line, skipping matches that start
    # inside an inline code span. Splitting the line at code spans is not
    # enough: an image/link whose *label* contains a code span (e.g.
    # ![`extract_table()` returns junk](assets/x.png)) straddles the split and
    # would never match. Span positions are recomputed per pass because
    # earlier rewrites shift offsets; code-span text itself is never modified.
    for regex, sub in (
        (MD_IMAGE_RE, image_sub),
        (MD_LINK_RE, link_sub),
        (HTML_SRC_RE, html_src_sub),
        (HTML_HREF_RE, html_href_sub),
    ):
        spans = [(m.start(), m.end()) for m in CODE_SPAN_RE.finditer(line)]

        def guarded(m: re.Match, _sub=sub, _spans=spans) -> str:
            if any(s <= m.start() < e for s, e in _spans):
                return m.group(0)
            return _sub(m)

        line = regex.sub(guarded, line)
    return line


def rewrite_links_and_images(lines: list[str], page_rel: str, ctx: StageContext) -> list[str]:
    out: list[str] = []
    tracker = FenceTracker()
    for line in lines:
        if tracker.feed(line):
            out.append(line)
            continue
        out.append(_transform_prose_line(line, page_rel, ctx))
    return out


# ---------------------------------------------------------------------------
# page compilation


def compile_page(page_rel: str, ctx: StageContext) -> None:
    authored_path = ctx.authored / page_rel
    executed_path = ctx.executed / page_rel

    meta, authored_body = split_frontmatter(authored_path.read_text(encoding="utf-8"))
    if executed_path.is_file():
        _, body_text = split_frontmatter(executed_path.read_text(encoding="utf-8"))
    else:
        body_text = authored_body

    lines = body_text.splitlines()
    lines = substitute_api_includes(lines, ctx, page_rel)
    lines = convert_admonitions(lines, ctx, page_rel)
    title, lines = lift_title(meta, lines, ctx, page_rel)
    lines = convert_tabs(lines, ctx, page_rel)
    lines = rewrite_links_and_images(lines, page_rel, ctx)
    lines = rewrite_output_fences(lines)

    if title is None:
        return  # error already collected

    out_meta: dict = {"title": title}
    if meta.get("description") is not None:
        out_meta["description"] = meta["description"]
    for key, value in meta.items():
        if key in EXECUTOR_KEYS or key in ("title", "description"):
            continue
        out_meta[key] = value

    fm = yaml.safe_dump(out_meta, sort_keys=False, allow_unicode=True, default_flow_style=False)
    body = "\n".join(lines).strip("\n")
    dest = ctx.content_dir / page_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# public files


def copy_public_tree(src: Path, dest_rel: str, ctx: StageContext, exclude_dotfiles: bool) -> None:
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if exclude_dotfiles and any(part.startswith(".") for part in rel.parts):
            continue
        target_rel = posixpath.join(dest_rel, rel.as_posix()) if dest_rel else rel.as_posix()
        dest = ctx.public_dir / target_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        ctx.copied_public.add(target_rel)


def copy_public_files(ctx: StageContext) -> None:
    llms = ctx.authored / "llms.txt"
    if llms.is_file():
        ctx.public_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(llms, ctx.public_dir / "llms.txt")
        ctx.copied_public.add("llms.txt")
    copy_public_tree(ctx.authored / "ocr-tool", "ocr-tool", ctx, exclude_dotfiles=True)
    copy_public_tree(ctx.authored / "assets", "assets", ctx, exclude_dotfiles=True)
    # Starlight's default favicon link points at /favicon.svg in public/.
    favicon = ctx.authored / "assets" / "favicon.svg"
    if favicon.is_file():
        shutil.copy2(favicon, ctx.public_dir / "favicon.svg")
        ctx.copied_public.add("favicon.svg")
    if ctx.executed.exists():
        for nb in sorted(ctx.executed.glob("*/notebooks/*.ipynb")):
            rel = nb.relative_to(ctx.executed).as_posix()
            dest = ctx.public_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nb, dest)
            ctx.copied_public.add(rel)


# ---------------------------------------------------------------------------
# discovery / validation


def discover_pages(ctx: StageContext) -> list[str]:
    pages = sorted(
        p.relative_to(ctx.authored).as_posix()
        for p in ctx.authored.rglob("*.md")
        if not p.relative_to(ctx.authored).as_posix().startswith("specs/")
    )
    ctx.pages = set(pages)
    if ctx.executed.exists():
        stale = sorted(
            p.relative_to(ctx.executed).as_posix()
            for p in ctx.executed.rglob("*.md")
            if p.relative_to(ctx.executed).as_posix() not in ctx.pages
        )
        if stale:
            listing = "\n".join(f"  - {ctx.executed / s}" for s in stale)
            ctx.error(
                "stale executed pages with no authored counterpart "
                f"(delete them or restore the authored source):\n{listing}"
            )
    return pages


def validate_site(site: Path, authored: Path, executed: Path) -> tuple[Path, Path]:
    site = site.resolve()
    authored = authored.resolve()
    executed = executed.resolve()
    content_dir = site / "src" / "content" / "docs"
    public_dir = site / "public"
    allowed_roots = {REPO_ROOT, authored.parent, executed.parent}
    for d in (content_dir, public_dir):
        if not any(_is_within(d, root) for root in allowed_roots):
            raise SystemExit(
                f"refusing --site {site}: {d} escapes the repository "
                f"(and the corpus directories)"
            )
        for protected in (authored, executed):
            if d == protected or _is_within(d, protected) or _is_within(protected, d):
                raise SystemExit(f"refusing --site {site}: {d} overlaps {protected}")
    return content_dir, public_dir


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


# ---------------------------------------------------------------------------
# main


def stage(
    authored: Path,
    executed: Path,
    site: Path,
    allow_missing_api: bool = False,
    fragments_dir: Path | None = None,
) -> tuple[int, int, list[str]]:
    """Run the staging pipeline. Returns (page_count, public_count, errors)."""
    if not authored.is_dir():
        raise SystemExit(f"authored docs directory not found: {authored}")
    content_dir, public_dir = validate_site(site, authored, executed)

    for d in (content_dir, public_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    ctx = StageContext(
        authored=authored.resolve(),
        executed=executed.resolve(),
        content_dir=content_dir,
        public_dir=public_dir,
        fragments_dir=(fragments_dir or REPO_ROOT / "temp" / "api-fragments").resolve(),
        allow_missing_api=allow_missing_api,
    )

    pages = discover_pages(ctx)
    for page_rel in pages:
        compile_page(page_rel, ctx)
    copy_public_files(ctx)
    return len(pages), len(ctx.copied_public), ctx.errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--authored", default="docs", type=Path)
    parser.add_argument("--executed", default="docs-executed", type=Path)
    parser.add_argument("--site", default="docs-site", type=Path)
    parser.add_argument(
        "--allow-missing-api",
        action="store_true",
        help="leave npdf-api:include markers in place when the fragment file is missing",
    )
    parser.add_argument(
        "--fragments",
        default=None,
        type=Path,
        help="API fragment directory (default: temp/api-fragments)",
    )
    args = parser.parse_args(argv)

    page_count, public_count, errors = stage(
        args.authored,
        args.executed,
        args.site,
        allow_missing_api=args.allow_missing_api,
        fragments_dir=args.fragments,
    )
    if errors:
        print(f"docs_stage: {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  * {err}", file=sys.stderr)
        return 1
    print(f"docs_stage: staged {page_count} pages, {public_count} public files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
