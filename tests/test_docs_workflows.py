from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_workflow_caches_match_pipeline_output_directory():
    for name in ("docs-fast.yml", "nightly-tutorials.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert '--out "docs-executed/$section"' in workflow
        assert "            docs-executed\n" in workflow
        assert "            docs-build\n" not in workflow


def test_nightly_uploads_the_directory_the_pipeline_produces():
    workflow = (REPO_ROOT / ".github" / "workflows" / "nightly-tutorials.yml").read_text(
        encoding="utf-8"
    )
    assert "          path: docs-executed\n" in workflow


def test_docs_pull_request_and_deploy_watch_pipeline_inputs():
    for name in ("docs-fast.yml", "docs.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for path in (
            "scripts/docs_build.py",
            "scripts/generate_reference.py",
            "scripts/docs_stage.py",
            "scripts/generate_api_reference.py",
            "docs-site/**",
            "pyproject.toml",
            "uv.lock",
        ):
            assert f"      - '{path}'\n" in workflow


def test_execution_cache_namespaces_and_inputs_are_isolated():
    expected = {
        "docs-fast.yml": "docs-executed-pr-v2-",
        "docs.yml": "docs-executed-deploy-v2-",
        "nightly-tutorials.yml": "docs-executed-nightly-v2-",
    }
    for name, namespace in expected.items():
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert namespace in workflow
        assert ".docs-build-cache.json" in workflow
        assert "'natural_pdf/**/*.py'" in workflow


def test_pdf_changes_trigger_docs_and_invalidate_workflow_caches():
    for name in ("docs-fast.yml", "docs.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "      - 'pdfs/**'\n" in workflow
    for name in ("docs-fast.yml", "docs.yml", "nightly-tutorials.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "'pdfs/**/*.pdf'" in workflow


def test_default_docs_cache_file_is_ignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".docs-build-cache.json" in gitignore.splitlines()


def test_all_execution_workflows_use_deploy_colab_layout():
    for name in ("docs-fast.yml", "docs.yml", "nightly-tutorials.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert '--colab-ref gh-pages --colab-prefix "$section"' in workflow


def test_pull_request_workflow_builds_and_verifies_starlight_site():
    workflow = (REPO_ROOT / ".github" / "workflows" / "docs-fast.yml").read_text(encoding="utf-8")
    assert "uv run --extra docs python scripts/generate_api_reference.py" in workflow
    assert "uv run --extra docs python scripts/docs_stage.py" in workflow
    assert "npm ci" in workflow
    assert "npm run build" in workflow
    assert "uv run --extra test pytest tests/test_docs_site_output.py" in workflow


def test_authored_docs_use_mkdocs_compatible_skip_fences_and_links():
    markdown = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / "docs").rglob("*.md"))
    )
    assert "```python skip=true" not in markdown
    assert "```python {.skip-execution}" in markdown

    quickstart = (REPO_ROOT / "docs" / "get-started" / "quickstart.md").read_text(encoding="utf-8")
    for target in ("../learn/", "../concepts/", "../solve/"):
        assert f"]({target})" not in quickstart
    pdfplumber = (REPO_ROOT / "docs" / "get-started" / "from-pdfplumber.md").read_text(
        encoding="utf-8"
    )
    assert "](../concepts/)" not in pdfplumber

    llms = (REPO_ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")
    assert "](/docs/" not in llms
    assert "https://jsoma.github.io/natural-pdf/" in llms


def test_solve_index_uses_staged_thumbnail_assets():
    # Cards are raw HTML (<a class="npdf-card"><img src="assets/...">) since
    # the Starlight conversion; the staging script rewrites the relative srcs.
    solve_index = (REPO_ROOT / "docs" / "solve" / "index.md").read_text(encoding="utf-8")
    assert solve_index.count('src="assets/') == 8
    assert "https://jsoma.github.io/natural-pdf/solve/assets/" not in solve_index
