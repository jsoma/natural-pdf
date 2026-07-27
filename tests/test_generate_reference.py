"""Tests for scripts/generate_reference.py — the docs reference generator."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_reference.py"

EXPECTED_PAGES = [
    "selectors.md",
    "engines.md",
    "installation-extras.md",
    "exceptions.md",
    "ocr-options.md",
]


@pytest.fixture(scope="module")
def genref():
    spec = importlib.util.spec_from_file_location("generate_reference", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("generate_reference", module)
    spec.loader.exec_module(module)
    return module


def test_generator_writes_all_pages(genref, tmp_path):
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0
    for name in EXPECTED_PAGES:
        path = out / name
        assert path.exists(), f"missing generated page {name}"
        content = path.read_text(encoding="utf-8")
        assert content.startswith(genref.HEADER_COMMENT), f"{name} missing header comment"
        assert content.endswith("\n")


def test_output_is_deterministic_across_runs(genref):
    first = genref.build_all_pages()
    second = genref.build_all_pages()
    assert first == second
    assert sorted(first) == sorted(EXPECTED_PAGES)


def test_check_passes_right_after_generation(genref, tmp_path):
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0
    assert genref.main(["--check", "--output", str(out)]) == 0


def test_check_fails_when_page_tampered(genref, tmp_path, capsys):
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0
    tampered = out / "selectors.md"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")
    assert genref.main(["--check", "--output", str(out)]) == 1
    err = capsys.readouterr().err
    assert "selectors.md" in err


def test_check_fails_when_page_missing(genref, tmp_path):
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0
    (out / "engines.md").unlink()
    assert genref.main(["--check", "--output", str(out)]) == 1


@pytest.fixture(scope="module")
def isolated_pages(genref):
    """One real isolated generation, shared by the orphan tests below."""
    return genref.build_all_pages_isolated()


def _use_cached_pages(genref, monkeypatch, isolated_pages):
    monkeypatch.setattr(genref, "build_all_pages_isolated", lambda: dict(isolated_pages))


def test_check_flags_orphan_and_regeneration_removes_it(
    genref, tmp_path, monkeypatch, isolated_pages, capsys
):
    _use_cached_pages(genref, monkeypatch, isolated_pages)
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0

    orphan = out / "obsolete-page.md"
    orphan.write_text(
        genref.HEADER_COMMENT + "\n\n# Obsolete\n\nOld generated content.\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    assert genref.main(["--check", "--output", str(out)]) == 1
    err = capsys.readouterr().err
    assert "obsolete-page.md" in err
    assert "orphaned" in err

    # Regeneration owns the header namespace: the orphan is deleted, loudly.
    assert genref.main(["--output", str(out)]) == 0
    assert not orphan.exists()
    assert "obsolete-page.md" in capsys.readouterr().out
    assert genref.main(["--check", "--output", str(out)]) == 0


def test_handwritten_page_without_header_never_touched_or_flagged(
    genref, tmp_path, monkeypatch, isolated_pages
):
    _use_cached_pages(genref, monkeypatch, isolated_pages)
    out = tmp_path / "reference"
    assert genref.main(["--output", str(out)]) == 0

    handwritten = out / "notes.md"
    content = "# Hand-written notes\n\nNot generated; no header.\n"
    handwritten.write_text(content, encoding="utf-8")

    assert genref.main(["--check", "--output", str(out)]) == 0
    assert genref.main(["--output", str(out)]) == 0
    assert handwritten.read_text(encoding="utf-8") == content


def test_find_orphaned_pages_only_matches_header_files(genref, tmp_path):
    out = tmp_path / "reference"
    out.mkdir()
    (out / "expected.md").write_text(genref.HEADER_COMMENT + "\n\n# E\n", encoding="utf-8")
    (out / "orphan.md").write_text(genref.HEADER_COMMENT + "\n\n# O\n", encoding="utf-8")
    (out / "manual.md").write_text("# Manual page\n", encoding="utf-8")
    orphans = genref.find_orphaned_pages(out, ["expected.md"])
    assert orphans == [out / "orphan.md"]


def test_committed_pages_are_up_to_date(genref):
    """CI drift gate: docs/reference in the repo must match a fresh generation."""
    assert genref.main(["--check"]) == 0


def test_selectors_page_covers_live_registry(genref):
    """Names on the page come from the live registry, not stale expectations."""
    import natural_pdf.selectors  # noqa: F401  (registers built-in clauses)
    from natural_pdf.selectors import parser as selector_parser
    from natural_pdf.selectors import registry as selector_registry

    page = genref.generate_selectors_page()

    # Every registered pseudo-class of every flavor must appear on the page.
    registered = (
        set(selector_registry._PSEUDO_HANDLERS)
        | set(selector_registry._POST_HANDLERS)
        | set(selector_registry._RELATIONAL_HANDLERS)
        | set(selector_parser._PSEUDO_CLASS_BUILDERS)
    )
    for name in registered:
        assert f"`:{name}`" in page, f"registered pseudo-class :{name} missing from page"

    for op in selector_parser._ATTRIBUTE_OP_BUILDERS:
        assert f"`{op}`" in page, f"attribute operator {op} missing from page"

    # Sanity-check a few names known to matter (and confirm they are real
    # registrations, so the assertion cannot pass vacuously).
    for known in ("contains", "bold", "regex"):
        assert known in selector_registry._PSEUDO_HANDLERS
        assert f"`:{known}`" in page


def test_selectors_drift_guard_fails_on_undescribed_registration(genref):
    """A registered name with no curated description must fail generation loudly."""
    from natural_pdf.selectors import registry as selector_registry

    name = "test-drift-guard-pseudo"
    selector_registry.register_pseudo(name, lambda pseudo, ctx: None, replace=True)
    try:
        with pytest.raises(genref.ReferenceGenerationError, match=name):
            genref.generate_selectors_page()
    finally:
        selector_registry.unregister_pseudo(name)


def test_check_is_immune_to_in_process_registry_mutation(genref):
    """The drift gate must not see registry mutations made by earlier tests.

    Selector clauses / engine registries are mutable process globals; other
    test modules (e.g. tests/test_engine_registry.py) register throwaway
    entries. main() introspects in a fresh subprocess, so a mutation in THIS
    process must not change the generated output.
    """
    from natural_pdf.selectors import registry as selector_registry

    name = "mutation-leak-check"
    selector_registry.register_pseudo(name, lambda pseudo, ctx: None, replace=True)
    try:
        assert genref.main(["--check"]) == 0
        # Sanity: the in-process (non-isolated) path DOES see the mutation,
        # proving the subprocess is what isolates the drift gate.
        with pytest.raises(genref.ReferenceGenerationError, match=name):
            genref.generate_selectors_page()
    finally:
        selector_registry.unregister_pseudo(name)


def test_introspect_json_mode_emits_all_pages(genref, tmp_path):
    """The self-invocation mode writes {page: markdown} JSON from a clean interpreter."""
    out = tmp_path / "pages.json"
    proc = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--introspect-json", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    pages = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(pages) == sorted(EXPECTED_PAGES)
    for name, content in pages.items():
        assert content.startswith(genref.HEADER_COMMENT), name


def test_curated_examples_all_parse(genref):
    """Every example selector destined for the page must be parser-valid."""
    from natural_pdf.selectors.parser import parse_selector

    examples = list(genref._iter_curated_examples())
    assert examples, "no curated examples collected"
    for origin, example in examples:
        parse_selector(example)  # raises on invalid syntax


def test_invalid_curated_example_fails_generation(genref, monkeypatch):
    """An example the parser rejects (numeric ~=) must fail generation loudly."""
    monkeypatch.setitem(
        genref.ATTRIBUTE_OP_DESCRIPTIONS, "~=", ("Approximately equal.", "text[size~=12]")
    )
    with pytest.raises(genref.ReferenceGenerationError, match="failed to parse"):
        genref.generate_selectors_page()


def test_generation_does_not_download_models(genref, monkeypatch):
    """Generation must never hit Hugging Face (or trigger any model download)."""
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def _blocked(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("network/model download attempted during generation")

    for attr in ("hf_hub_download", "snapshot_download", "hf_hub_url"):
        if hasattr(huggingface_hub, attr):
            monkeypatch.setattr(huggingface_hub, attr, _blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    pages = genref.build_all_pages()
    assert sorted(pages) == sorted(EXPECTED_PAGES)


def test_engines_page_lists_registered_capabilities(genref):
    genref._import_engine_provider_modules()
    from natural_pdf.engine_registry import list_engines

    page = genref.generate_engines_page()
    for capability, names in list_engines().items():
        assert f"`{capability}`" in page, f"capability {capability} missing"
        for name in names:
            assert f"`{name}`" in page, f"engine {name} ({capability}) missing"


def test_extras_page_lists_all_groups_and_deps(genref):
    from natural_pdf.utils.optional_imports import (
        OPTIONAL_DEPENDENCIES,
        OPTIONAL_DEPENDENCY_GROUPS,
    )

    page = genref.generate_installation_extras_page()
    for group in OPTIONAL_DEPENDENCY_GROUPS:
        assert f"natural-pdf[{group}]" in page
    for dep_name in OPTIONAL_DEPENDENCIES:
        assert f"`{dep_name}`" in page


def test_exceptions_page_matches_module(genref):
    import inspect

    import natural_pdf.exceptions as exc_module

    page = genref.generate_exceptions_page()
    for name, obj in vars(exc_module).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseException)
            and obj.__module__ == exc_module.__name__
        ):
            assert f"`{name}" in page, f"exception {name} missing from page"


def test_ocr_options_page_covers_option_classes(genref):
    import dataclasses
    import inspect

    from natural_pdf.ocr import ocr_options as options_module

    page = genref.generate_ocr_options_page()
    for name, obj in vars(options_module).items():
        if (
            inspect.isclass(obj)
            and dataclasses.is_dataclass(obj)
            and issubclass(obj, options_module.BaseOCROptions)
            and obj.__module__ == options_module.__name__
        ):
            assert f"`{name}`" in page, f"options class {name} missing from page"
    # Spot-check field rows exist with defaults.
    assert "`strip_math`" in page
    assert "`extra_args`" in page
