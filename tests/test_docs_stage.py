"""Tests for scripts/docs_stage.py using synthetic corpora in tmp_path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import docs_stage  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    authored = tmp_path / "docs"
    executed = tmp_path / "docs-executed"
    site = tmp_path / "site"
    frags = tmp_path / "frags"
    for d in (authored, executed, frags):
        d.mkdir()
    return authored, executed, site, frags


def run_stage(corpus, allow_missing_api=False):
    authored, executed, site, frags = corpus
    return docs_stage.stage(
        authored, executed, site, allow_missing_api=allow_missing_api, fragments_dir=frags
    )


def staged(corpus, rel):
    _, _, site, _ = corpus
    return (site / "src" / "content" / "docs" / rel).read_text(encoding="utf-8")


def write(base: Path, rel: str, text: str):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# titles / frontmatter


def test_title_lifted_from_h1_and_removed(corpus):
    authored = corpus[0]
    write(authored, "page.md", "# My Title\n\nBody text.\n")
    pages, _, errors = run_stage(corpus)
    assert errors == []
    assert pages == 1
    out = staged(corpus, "page.md")
    assert "title: My Title" in out
    assert "# My Title" not in out
    assert "Body text." in out


def test_explicit_title_used_and_matching_h1_removed(corpus):
    authored = corpus[0]
    write(authored, "page.md", '---\ntitle: "My Title"\n---\n\n# My Title\n\nBody.\n')
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "title: My Title" in out
    assert "# My Title" not in out


def test_conflicting_title_and_h1_errors(corpus):
    authored = corpus[0]
    write(authored, "page.md", "---\ntitle: One Thing\n---\n\n# Another Thing\n\nBody.\n")
    _, _, errors = run_stage(corpus)
    assert any("conflicts" in e for e in errors)


def test_no_derivable_title_errors(corpus):
    authored = corpus[0]
    write(authored, "page.md", "Just prose, no heading.\n")
    _, _, errors = run_stage(corpus)
    assert any("no title" in e for e in errors)


def test_h1_inside_code_fence_not_used_as_title(corpus):
    authored = corpus[0]
    write(authored, "page.md", "```\n# not a title\n```\n\n# Real Title\n\nBody.\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "title: Real Title" in out
    assert "# not a title" in out  # untouched inside fence


def test_executor_keys_dropped_description_kept(corpus):
    authored = corpus[0]
    write(
        authored,
        "page.md",
        "---\nfixture: pdfs/x.pdf\ntier: 2\nthumbnail: 1\nskip: true\n"
        "description: A nice page.\n---\n\n# Title\n\nBody.\n",
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    fm = out.split("---")[1]
    assert "description: A nice page." in fm
    for key in ("fixture", "tier", "thumbnail", "skip"):
        assert key not in fm


# ---------------------------------------------------------------------------
# overlay / stale


def test_executed_body_wins_metadata_from_authored(corpus):
    authored, executed = corpus[0], corpus[1]
    write(
        authored,
        "page.md",
        "---\ndescription: Desc.\nfixture: f.pdf\n---\n\n# Title\n\nauthored body\n",
    )
    write(executed, "page.md", "# Title\n\nexecuted body\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "executed body" in out
    assert "authored body" not in out
    assert "description: Desc." in out


def test_stale_executed_page_errors(corpus):
    authored, executed = corpus[0], corpus[1]
    write(authored, "page.md", "# Title\n\nBody.\n")
    write(executed, "gone.md", "# Gone\n\nStale.\n")
    _, _, errors = run_stage(corpus)
    assert any("stale" in e and "gone.md" in e for e in errors)


def test_specs_excluded(corpus):
    authored = corpus[0]
    write(authored, "page.md", "# Title\n\nBody.\n")
    write(authored, "specs/spec.md", "no title here\n")
    pages, _, errors = run_stage(corpus)
    assert errors == []
    assert pages == 1


# ---------------------------------------------------------------------------
# tabs


def test_tab_group_conversion(corpus):
    authored = corpus[0]
    write(
        authored,
        "page.md",
        "# Title\n\n"
        "/// tab | pdfplumber\n\n"
        "```python\nx = 1\n```\n\n"
        "///\n\n"
        "/// tab | Natural PDF\n\n"
        "second body\n\n"
        "///\n\n"
        "After tabs.\n",
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert out.count("<!-- npdf-tabs:start -->") == 1
    assert out.count("<!-- npdf-tabs:end -->") == 1
    assert '<!-- npdf-tab:start label="pdfplumber" -->' in out
    assert '<!-- npdf-tab:start label="Natural PDF" -->' in out
    assert out.count("<!-- npdf-tab:end -->") == 2
    # fenced code inside tab body untouched
    assert "```python\nx = 1\n```" in out
    assert "After tabs." in out
    # group markers ordered correctly
    assert out.index("npdf-tabs:start") < out.index('label="pdfplumber"')
    assert out.index('label="Natural PDF"') < out.index("npdf-tabs:end")


def test_separated_tab_blocks_form_two_groups(corpus):
    authored = corpus[0]
    write(
        authored,
        "page.md",
        "# Title\n\n/// tab | A\n\nbody a\n\n///\n\nIntervening prose.\n\n"
        "/// tab | B\n\nbody b\n\n///\n",
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert out.count("<!-- npdf-tabs:start -->") == 2


def test_tab_terminator_inside_fence_ignored(corpus):
    authored = corpus[0]
    write(
        authored,
        "page.md",
        "# Title\n\n/// tab | A\n\n```\n///\n```\n\nstill inside tab\n\n///\n",
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "still inside tab" in out
    assert out.count("<!-- npdf-tab:end -->") == 1
    assert "```\n///\n```" in out


# ---------------------------------------------------------------------------
# fences


def test_output_fence_becomes_shaded_pre(corpus):
    authored = corpus[0]
    write(authored, "page.md", "# Title\n\n```output\nhello\n```\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert '<pre class="npdf-output"><code>hello</code></pre>' in out
    assert "```output" not in out


def test_output_fence_content_html_escaped_and_not_link_rewritten(corpus):
    authored = corpus[0]
    write(
        authored,
        "page.md",
        '# Title\n\n```output\n<Page 1> & src="x.png" [link](other.md)\n```\n',
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "&lt;Page 1&gt; &amp;" in out
    # raw output is display text: no attribute or link rewriting applies
    assert 'src="x.png"' in out
    assert "[link](other.md)" in out


def test_output_word_inside_fence_untouched(corpus):
    authored = corpus[0]
    write(authored, "page.md", "# Title\n\n```python\nprint('```output')\n```\n")
    _, _, errors = run_stage(corpus)
    assert errors == []


# ---------------------------------------------------------------------------
# links


def link_page(corpus, rel, body):
    write(corpus[0], rel, f"# T {rel}\n\n{body}\n")


def test_link_rewrites(corpus):
    authored = corpus[0]
    link_page(corpus, "a/b.md", "[sib](c.md) [dotsib](./c.md) [up](../top.md)")
    link_page(corpus, "a/c.md", "[to b](b.md)")
    link_page(corpus, "a/index.md", "[child](c.md)")
    link_page(corpus, "top.md", "[down](a/c.md) [idx](a/index.md)")
    link_page(corpus, "index.md", "[root down](a/c.md) [root top](top.md)")
    _, _, errors = run_stage(corpus)
    assert errors == []
    b = staged(corpus, "a/b.md")
    assert "[sib](../c/)" in b
    assert "[dotsib](../c/)" in b
    assert "[up](../../top/)" in b
    idx = staged(corpus, "a/index.md")
    assert "[child](c/)" in idx
    # top.md itself becomes the /top/ directory, so its links need ../
    top = staged(corpus, "top.md")
    assert "[down](../a/c/)" in top
    assert "[idx](../a/)" in top
    # the root index.md maps to the site root, so its links are bare paths
    root = staged(corpus, "index.md")
    assert "[root down](a/c/)" in root
    assert "[root top](top/)" in root


def test_link_fragment_preserved(corpus):
    link_page(corpus, "a/b.md", "[frag](c.md#some-section)")
    link_page(corpus, "a/c.md", "text")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert "[frag](../c/#some-section)" in staged(corpus, "a/b.md")


def test_external_anchor_mailto_links_untouched(corpus):
    link_page(
        corpus,
        "page.md",
        "[ext](https://example.com/x.md) [mail](mailto:a@b.com) [anch](#here) [abs](/x.md)",
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "page.md")
    assert "(https://example.com/x.md)" in out
    assert "(mailto:a@b.com)" in out
    assert "(#here)" in out
    assert "(/x.md)" in out


def test_raw_html_href_rewritten(corpus):
    link_page(corpus, "a/b.md", '<a href="c.md#x">go</a>')
    link_page(corpus, "a/c.md", "text")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert '<a href="../c/#x">go</a>' in staged(corpus, "a/b.md")


def test_missing_link_target_errors(corpus):
    link_page(corpus, "page.md", "[bad](missing.md)")
    _, _, errors = run_stage(corpus)
    assert any("link target not found" in e for e in errors)


def test_links_inside_fence_untouched(corpus):
    link_page(corpus, "a/b.md", "```\n[x](c.md)\n```")
    link_page(corpus, "a/c.md", "text")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert "[x](c.md)" in staged(corpus, "a/b.md")


# ---------------------------------------------------------------------------
# images


def test_markdown_image_copied_and_rewritten(corpus):
    authored, executed, site, _ = corpus
    write(authored, "solve/index.md", "# Solve\n\n![shot](assets/x.png)\n")
    img = executed / "solve" / "assets" / "x.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\x89PNG fake")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert "![shot](/natural-pdf/solve/assets/x.png)" in staged(corpus, "solve/index.md")
    assert (site / "public" / "solve" / "assets" / "x.png").read_bytes() == b"\x89PNG fake"


def test_authored_image_takes_precedence(corpus):
    authored, executed, site, _ = corpus
    write(authored, "p.md", "# T\n\n![i](img.png)\n")
    (authored / "img.png").write_bytes(b"authored")
    (executed / "img.png").write_bytes(b"executed")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert (site / "public" / "img.png").read_bytes() == b"authored"


def test_raw_html_src_copied_and_rewritten(corpus):
    authored, _, site, _ = corpus
    write(authored, "p.md", '# T\n\n<img src="assets/pic.png" alt="x">\n')
    pic = authored / "assets" / "pic.png"
    pic.parent.mkdir()
    pic.write_bytes(b"pngpng")
    _, _, errors = run_stage(corpus)
    assert errors == []
    assert '<img src="/natural-pdf/assets/pic.png"' in staged(corpus, "p.md")
    assert (site / "public" / "assets" / "pic.png").is_file()


def test_missing_image_errors(corpus):
    write(corpus[0], "p.md", "# T\n\n![gone](nope.png)\n")
    _, _, errors = run_stage(corpus)
    assert any("image not found" in e for e in errors)


# ---------------------------------------------------------------------------
# admonitions / asides


def test_legacy_admonition_converted(corpus):
    write(
        corpus[0],
        "p.md",
        '# T\n\n!!! warning "Heads up"\n    First line.\n\n    Second line.\n\nAfter.\n',
    )
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "p.md")
    assert ":::caution[Heads up]" in out
    assert "\nFirst line.\n" in out
    assert "\nSecond line.\n" in out
    assert "!!!" not in out
    assert "After." in out


def test_starlight_aside_passes_through(corpus):
    write(corpus[0], "p.md", "# T\n\n:::caution[Watch out]\nBody.\n:::\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "p.md")
    assert ":::caution[Watch out]\nBody.\n:::" in out


# ---------------------------------------------------------------------------
# public files


def test_public_files_copied_ocr_tool_dotfiles_excluded(corpus):
    authored, executed, site, _ = corpus
    write(authored, "p.md", "# T\n\nBody.\n")
    (authored / "llms.txt").write_text("llms\n")
    write(authored, "ocr-tool/index.html", "<html></html>")
    write(authored, "ocr-tool/js/app.js", "//js")
    write(authored, "ocr-tool/.DS_Store", "junk")
    write(authored, "assets/logo.svg", "<svg/>")
    nb = executed / "learn" / "notebooks" / "01.ipynb"
    nb.parent.mkdir(parents=True)
    nb.write_text("{}")
    _, public_count, errors = run_stage(corpus)
    assert errors == []
    pub = site / "public"
    assert (pub / "llms.txt").is_file()
    assert (pub / "ocr-tool" / "index.html").is_file()
    assert (pub / "ocr-tool" / "js" / "app.js").is_file()
    assert not (pub / "ocr-tool" / ".DS_Store").exists()
    assert (pub / "assets" / "logo.svg").is_file()
    assert (pub / "learn" / "notebooks" / "01.ipynb").is_file()
    assert public_count == 5


# ---------------------------------------------------------------------------
# api include markers


def test_api_include_substituted(corpus):
    frags = corpus[3]
    (frags / "pdf-class.md").write_text("## PDF\n\nfragment body\n")
    write(corpus[0], "api.md", "# API\n\n<!-- npdf-api:include id=pdf-class -->\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "api.md")
    assert "fragment body" in out
    assert "npdf-api:include" not in out


def test_api_include_missing_errors(corpus):
    write(corpus[0], "api.md", "# API\n\n<!-- npdf-api:include id=nope -->\n")
    _, _, errors = run_stage(corpus)
    assert any("missing API fragment" in e for e in errors)


def test_api_include_missing_allowed_leaves_marker(corpus):
    write(corpus[0], "api.md", "# API\n\n<!-- npdf-api:include id=nope -->\n")
    _, _, errors = run_stage(corpus, allow_missing_api=True)
    assert errors == []
    assert "<!-- npdf-api:include id=nope -->" in staged(corpus, "api.md")


# ---------------------------------------------------------------------------
# fence safety (catch-all)


def test_no_transformation_inside_code_fences(corpus):
    write(
        corpus[0],
        "a/b.md",
        '# T\n\n```text\n/// tab | Nope\n!!! warning "No"\n[x](c.md)\n![y](img.png)\n///\n```\n',
    )
    write(corpus[0], "a/c.md", "# C\n\ntext\n")
    _, _, errors = run_stage(corpus)
    assert errors == []
    out = staged(corpus, "a/b.md")
    assert "/// tab | Nope" in out
    assert '!!! warning "No"' in out
    assert "[x](c.md)" in out
    assert "![y](img.png)" in out
    assert "npdf-tab" not in out


# ---------------------------------------------------------------------------
# site path safety


def test_refuses_site_equal_to_authored(corpus):
    authored, executed, _, frags = corpus
    write(authored, "p.md", "# T\n\nBody.\n")
    with pytest.raises(SystemExit):
        docs_stage.stage(authored, executed, authored, fragments_dir=frags)


def test_refuses_site_escaping_allowed_roots(corpus):
    authored, executed, _, frags = corpus
    write(authored, "p.md", "# T\n\nBody.\n")
    with pytest.raises(SystemExit):
        docs_stage.stage(authored, executed, Path("/"), fragments_dir=frags)


# ---------------------------------------------------------------------------
# inline code spans vs. link/image rewriting


def test_image_with_code_span_in_alt_is_rewritten(corpus):
    authored, _, _, _ = corpus
    write(authored, "assets/pic.png", "png-bytes")
    write(
        authored,
        "trouble.md",
        "# T\n\n![`extract_table()` returns junk or nothing](assets/pic.png)\n",
    )
    run_stage(corpus)
    out = staged(corpus, "trouble.md")
    assert "![`extract_table()` returns junk or nothing](/natural-pdf/assets/pic.png)" in out


def test_link_inside_code_span_untouched(corpus):
    authored, _, _, _ = corpus
    write(authored, "other.md", "# Other\n\nBody.\n")
    write(authored, "p.md", "# T\n\nUse `[x](other.md)` literally, or see [x](other.md).\n")
    run_stage(corpus)
    out = staged(corpus, "p.md")
    assert "`[x](other.md)`" in out
    assert "[x](../other/)" in out
