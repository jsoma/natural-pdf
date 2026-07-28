"""Verifier for the built Starlight docs site (docs-site/dist).

Implements section 8 ("Verifier") of docs/specs/starlight_migration_spec.md.
Runs against the static output tree only — no server, no network.
"""

import json
import re
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "docs-site" / "dist"
DOCS = REPO_ROOT / "docs"
EXECUTED = REPO_ROOT / "docs-executed"
ASTRO_CONFIG = REPO_ROOT / "docs-site" / "astro.config.mjs"

BASE = "/natural-pdf"
SITE_URL = "https://jsoma.github.io/natural-pdf/"

if not (DIST / "index.html").exists():
    pytest.skip("docs site not built", allow_module_level=True)


# --- helpers -----------------------------------------------------------------


def expected_routes():
    """Every authored page maps to a route: index.md -> its directory, page.md -> page/."""
    routes = []
    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(DOCS)
        if rel.parts[0] == "specs":
            continue
        if rel.name == "index.md":
            route = "/".join(rel.parent.parts)
        else:
            route = "/".join(rel.with_suffix("").parts)
        routes.append(route)  # "" is the site root
    return routes


ROUTES = expected_routes()


def is_redirect(html):
    return 'http-equiv="refresh"' in html


def all_html_pages():
    return sorted(DIST.rglob("*.html"))


def canonical_pages():
    return [p for p in all_html_pages() if not is_redirect(p.read_text(encoding="utf-8"))]


def parse_config_redirects():
    """Textual parse of the redirects object in astro.config.mjs (the source of truth)."""
    text = ASTRO_CONFIG.read_text(encoding="utf-8")
    match = re.search(r"redirects:\s*\{(.*?)\n\s*\}", text, re.S)
    assert match, "no redirects object found in astro.config.mjs"
    return re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', match.group(1))


class _LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if not value:
                continue
            if name in ("href", "src"):
                self.targets.append(value)
            elif name == "srcset":
                for candidate in value.split(","):
                    parts = candidate.strip().split()
                    if parts:
                        self.targets.append(parts[0])


def main_content(html):
    match = re.search(r"<main[^>]*>(.*)</main>", html, re.S)
    return match.group(1) if match else html


# --- page routes -------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES, ids=[r or "(root)" for r in ROUTES])
def test_route_page_exists_with_single_h1(route):
    page = DIST / route / "index.html" if route else DIST / "index.html"
    assert page.exists(), f"missing page for route '/{route}/': {page}"
    html = page.read_text(encoding="utf-8")
    h1_count = len(re.findall(r"<h1[\s>]", main_content(html)))
    assert (
        h1_count == 1
    ), f"route '/{route}/' has {h1_count} <h1> tags in <main>, expected exactly 1"


# --- redirects ---------------------------------------------------------------


def test_redirects_match_config_and_resolve():
    redirects = parse_config_redirects()
    assert (
        len(redirects) == 59
    ), f"expected 59 redirects in astro.config.mjs, found {len(redirects)}"
    failures = []
    for source, dest in redirects:
        page = DIST / source.strip("/") / "index.html"
        if not page.exists():
            failures.append(f"{source}: no dist page at {page}")
            continue
        html = page.read_text(encoding="utf-8")
        if 'http-equiv="refresh"' not in html:
            failures.append(f"{source}: page is not a meta-refresh redirect")
        if '<meta name="robots" content="noindex">' not in html:
            failures.append(f"{source}: missing robots noindex meta")
        if not dest.startswith(BASE + "/"):
            failures.append(f"{source}: destination {dest} does not start with {BASE}/")
            continue
        if f'url={dest}"' not in html:
            failures.append(f"{source}: page does not refresh to {dest}")
        target = DIST / dest[len(BASE) :].strip("/") / "index.html"
        if not target.exists():
            failures.append(f"{source}: destination {dest} has no page at {target}")
        elif is_redirect(target.read_text(encoding="utf-8")):
            failures.append(f"{source}: destination {dest} is itself a redirect")
    assert not failures, "redirect problems:\n" + "\n".join(failures)


# --- link integrity ----------------------------------------------------------


def test_internal_links_resolve():
    failures = []
    for page in canonical_pages():
        parser = _LinkCollector()
        parser.feed(page.read_text(encoding="utf-8"))
        rel_page = page.relative_to(DIST)
        for raw in parser.targets:
            url = raw.split("#", 1)[0]
            if not url:
                continue  # pure fragment
            split = urllib.parse.urlsplit(url)
            if split.scheme or url.startswith("//"):
                continue  # http(s), mailto:, tel:, data:, protocol-relative
            if url.endswith(".md"):
                failures.append(f"{rel_page}: internal link ends in .md: {raw}")
                continue
            if url.startswith("/"):
                if url != BASE + "/" and not url.startswith(BASE + "/"):
                    failures.append(f"{rel_page}: root-absolute link missing {BASE} prefix: {raw}")
                    continue
                target = DIST / urllib.parse.unquote(url[len(BASE) :].lstrip("/"))
            else:
                target = (page.parent / urllib.parse.unquote(url)).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                failures.append(f"{rel_page}: broken internal link: {raw}")
    assert not failures, f"{len(failures)} link problems:\n" + "\n".join(failures)


# --- legacy markers ----------------------------------------------------------


LEGACY_MARKERS = [
    "/// tab",
    "!!! warning",
    "npdf-tabs:start",
    "npdf-api:include",
    "::: natural_pdf",
]


def test_no_legacy_markers_in_output():
    failures = []
    for page in all_html_pages():
        html = page.read_text(encoding="utf-8")
        for marker in LEGACY_MARKERS:
            if marker in html:
                failures.append(f"{page.relative_to(DIST)}: contains literal {marker!r}")
    assert not failures, "legacy markers leaked into output:\n" + "\n".join(failures)


# --- public contract files ---------------------------------------------------


def test_public_contract_files_exist():
    expected = [
        "llms.txt",
        "favicon.svg",
        # Without .nojekyll, GitHub Pages' Jekyll pass drops the _astro/
        # asset directory and the site serves completely unstyled.
        ".nojekyll",
        "404.html",
        "ocr-tool/index.html",
        "ocr-tool/words.txt",
        "ocr-tool/css/style.css",
        "ocr-tool/js/app.js",
    ]
    missing = [rel for rel in expected if not (DIST / rel).is_file()]
    assert not missing, f"missing public contract files: {missing}"
    vendor = DIST / "ocr-tool" / "js" / "vendor"
    assert vendor.is_dir() and any(vendor.iterdir()), "ocr-tool/js/vendor is missing or empty"


EXPECTED_NOTEBOOKS = sorted(
    str(p.relative_to(EXECUTED)) for p in EXECUTED.glob("*/notebooks/*.ipynb")
)


def test_executed_notebooks_are_published():
    assert EXPECTED_NOTEBOOKS, f"no notebooks found under {EXECUTED} — executor output missing"
    failures = []
    for rel in EXPECTED_NOTEBOOKS:
        published = DIST / rel
        if not published.is_file():
            failures.append(f"{rel}: not published to dist")
            continue
        try:
            json.loads(published.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{rel}: published notebook is not valid JSON ({exc})")
    assert not failures, "notebook problems:\n" + "\n".join(failures)


def test_nothing_from_specs_published():
    leaked = [str(p.relative_to(DIST)) for p in DIST.rglob("*") if "specs" in p.parts]
    assert not leaked, f"specs content leaked into dist: {leaked}"


# --- sitemap -----------------------------------------------------------------


def test_sitemap():
    index = DIST / "sitemap-index.xml"
    assert index.exists(), "sitemap-index.xml missing"
    sitemap_urls = re.findall(r"<loc>([^<]+)</loc>", index.read_text(encoding="utf-8"))
    assert sitemap_urls, "sitemap-index.xml references no sitemaps"
    failures = []
    for sitemap_url in sitemap_urls:
        name = sitemap_url.rsplit("/", 1)[-1]
        sitemap = DIST / name
        if not sitemap.exists():
            failures.append(f"referenced sitemap {name} missing from dist")
            continue
        for loc in re.findall(r"<loc>([^<]+)</loc>", sitemap.read_text(encoding="utf-8")):
            if not loc.startswith(SITE_URL):
                failures.append(f"{name}: URL outside site base: {loc}")
            if not loc.endswith("/"):
                failures.append(f"{name}: URL missing trailing slash: {loc}")
    assert not failures, "sitemap problems:\n" + "\n".join(failures)


# --- API reference -----------------------------------------------------------


def test_api_page_anchors():
    html = (DIST / "api" / "index.html").read_text(encoding="utf-8")
    assert '<a id="natural_pdf.PDF">' in html, "API page missing anchor for natural_pdf.PDF"
    anchor_count = len(re.findall(r'id="natural_pdf\.', html))
    assert (
        anchor_count >= 500
    ), f"API page has only {anchor_count} natural_pdf.* anchors (expected >= 500)"


# --- search ------------------------------------------------------------------


def test_pagefind_index_exists():
    pagefind = DIST / "pagefind"
    assert pagefind.is_dir(), "pagefind/ directory missing from dist"
    assert (pagefind / "pagefind-entry.json").is_file(), "pagefind-entry.json missing"
    index_files = list(pagefind.glob("index/*.pf_index"))
    assert index_files, "no .pf_index files under pagefind/index/"
