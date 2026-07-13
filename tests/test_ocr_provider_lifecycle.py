"""Regressions for OCR dispatch through EngineProvider lifecycle ownership."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from PIL import Image

import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.engine_registry import register_ocr_engine
from natural_pdf.ocr.ocr_options import BaseOCROptions
from natural_pdf.ocr.ocr_provider import cleanup_engine, run_ocr_apply, run_ocr_engine
from natural_pdf.ocr.unified_dispatch import (
    EngineCache,
    EngineEntry,
    get_registry,
    register_engine,
    run_ocr,
)


class _Target:
    def render(self, resolution=72, **kwargs):
        return Image.new("RGB", (8, 8), "white")


@pytest.fixture
def isolated_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)
    registry = get_registry()
    previous_registry = dict(registry)
    yield provider
    provider.clear()
    registry.clear()
    registry.update(previous_registry)


@dataclass
class _CustomOptions(BaseOCROptions):
    profile: str = "default"


def test_public_custom_engine_uses_provider_context_and_constructor_variants(
    isolated_provider,
):
    name = f"ocr.lifecycle.{uuid.uuid4().hex}"
    constructed = []

    class Engine:
        def __init__(self, serial):
            self.serial = serial

        def process_image(self, image, **kwargs):
            return [
                {
                    "bbox": [0, 0, 1, 1],
                    "text": str(self.serial),
                    "confidence": 1.0,
                }
            ]

    def factory(*, context, options, languages, device):
        engine = Engine(len(constructed))
        constructed.append((context, options, languages, device, engine))
        return engine

    register_ocr_engine(name, factory, options_class=_CustomOptions)
    assert get_registry()[name].engine_type == "classic_provider"

    first_context = object()
    second_context = object()
    options = _CustomOptions(profile="accurate")

    first = run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["en"],
        device="cpu",
        options=options,
    )
    repeated = run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["en"],
        device="cpu",
        options=options,
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=second_context,
        languages=["en"],
        device="cpu",
        options=options,
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["fr"],
        device="cpu",
        options=options,
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["en"],
        device="mps",
        options=options,
    )

    assert repeated.results[0]["text"] == first.results[0]["text"]
    assert len(constructed) == 4
    assert constructed[0][:4] == (first_context, options, ["en"], "cpu")

    # EngineProvider documents opaque constructor objects as identity-keyed.
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["en"],
        device="cpu",
        options=_CustomOptions(profile="accurate"),
    )
    assert len(constructed) == 5


def test_provider_only_factory_receives_normalized_options_without_extra_kwargs(
    isolated_provider,
):
    name = f"ocr.provider-only.{uuid.uuid4().hex}"
    constructed = []

    class ProviderOptions(BaseOCROptions):
        pass

    class Engine:
        def process_image(self, image, **kwargs):
            return []

    def factory(*, context, options):
        engine = Engine()
        constructed.append((context, options, engine))
        return engine

    isolated_provider.register("ocr.apply", name, factory)
    context = object()
    options = ProviderOptions()

    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=context,
        languages=["en"],
        device="cpu",
        options=options,
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=context,
        languages=["fr"],
        device="mps",
        options=options,
    )
    assert len(constructed) == 1
    assert constructed[0][:2] == (context, options)

    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=context,
        options=ProviderOptions(),
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=object(),
        options=options,
    )
    assert len(constructed) == 3


def test_public_custom_provider_initializes_each_model_variant_once(isolated_provider):
    name = f"ocr.initialize.{uuid.uuid4().hex}"
    initialized = []

    class Engine:
        def __init__(self):
            self._initialized = False

        def _initialize_model(self, languages, device, options):
            initialized.append((self, languages, device, options))

        def process_image(self, image, **kwargs):
            assert self._initialized
            return []

    register_ocr_engine(name, Engine, options_class=_CustomOptions)
    context = object()
    options = _CustomOptions(profile="models")

    for _ in range(2):
        run_ocr(
            target=_Target(),
            engine_name=name,
            resolution=72,
            context=context,
            languages=["en"],
            device="cpu",
            options=options,
        )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=context,
        languages=["fr"],
        device="cpu",
        options=options,
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=context,
        languages=["en"],
        device="mps",
        options=options,
    )

    assert len(initialized) == 3
    assert [record[1:3] for record in initialized] == [
        (["en"], "cpu"),
        (["fr"], "cpu"),
        (["en"], "mps"),
    ]
    assert all(record[3] is options for record in initialized)
    assert len({id(record[0]) for record in initialized}) == 3


def test_unavailable_public_custom_provider_is_not_cached(isolated_provider):
    name = f"ocr.unavailable.{uuid.uuid4().hex}"
    constructed = []

    class Engine:
        def is_available(self):
            return False

        def _initialize_model(self, languages, device, options):  # pragma: no cover
            raise AssertionError("unavailable engines must not initialize")

        def process_image(self, image, **kwargs):  # pragma: no cover
            raise AssertionError("unavailable engines must not run")

    def factory():
        engine = Engine()
        constructed.append(engine)
        return engine

    register_ocr_engine(name, factory, install_hint="install custom-ocr")
    for _ in range(2):
        with pytest.raises(
            RuntimeError,
            match="Install it with: install custom-ocr",
        ):
            run_ocr(
                target=_Target(),
                engine_name=name,
                resolution=72,
                context=None,
                device="cpu",
            )

    assert len(constructed) == 2


def test_legacy_provider_paths_forward_options(isolated_provider):
    apply_name = f"ocr.legacy-apply.{uuid.uuid4().hex}"
    extract_name = f"ocr.legacy-extract.{uuid.uuid4().hex}"
    received = []

    class ProviderOptions(BaseOCROptions):
        pass

    class Engine:
        def process_image(self, image=None, images=None, **kwargs):
            return []

    def factory(*, context, options):
        received.append((context, options))
        return Engine()

    isolated_provider.register("ocr.apply", apply_name, factory)
    isolated_provider.register("ocr.extract", extract_name, factory)
    apply_context = object()
    extract_context = object()
    apply_options = ProviderOptions()
    extract_options = ProviderOptions()

    run_ocr_apply(
        target=_Target(),
        context=apply_context,
        engine_name=apply_name,
        resolution=72,
        options=apply_options,
    )
    run_ocr_engine(
        Image.new("RGB", (8, 8), "white"),
        context=extract_context,
        engine_name=extract_name,
        options=extract_options,
    )

    assert received == [(apply_context, apply_options), (extract_context, extract_options)]


def test_public_registration_replace_and_cleanup_are_provider_owned(isolated_provider):
    name = f"ocr.replace.{uuid.uuid4().hex}"
    cleaned = []

    class Engine:
        def __init__(self, label):
            self.label = label

        def process_image(self, image, **kwargs):
            return [
                {
                    "bbox": [0, 0, 1, 1],
                    "text": self.label,
                    "confidence": 1.0,
                }
            ]

        def cleanup(self):
            cleaned.append(self.label)

    register_ocr_engine(
        name,
        lambda: Engine("first"),
        install_hint="install first",
        metadata={"version": 1},
    )
    context = object()
    first = run_ocr(
        target=_Target(), engine_name=name, resolution=72, context=context, device="cpu"
    )
    original_entry = get_registry()[name]

    register_ocr_engine(
        name,
        lambda: Engine("skipped"),
        install_hint="install skipped",
        metadata={"version": 2},
        replace=False,
    )
    assert get_registry()[name] is original_entry
    assert isolated_provider.get_metadata("ocr.apply", name) == {
        "version": 1,
        "install_hint": "install first",
        "kind": "classic",
    }
    repeated = run_ocr(
        target=_Target(), engine_name=name, resolution=72, context=context, device="cpu"
    )
    assert first.results[0]["text"] == repeated.results[0]["text"] == "first"

    register_ocr_engine(name, lambda: Engine("replacement"), metadata={"version": 3})
    assert cleaned == ["first"]
    replacement = run_ocr(
        target=_Target(), engine_name=name, resolution=72, context=context, device="cpu"
    )
    assert replacement.results[0]["text"] == "replacement"

    assert cleanup_engine(name) == 1
    assert cleaned == ["first", "replacement"]


def test_cleanup_all_includes_public_custom_provider_variants(isolated_provider, monkeypatch):
    name = f"ocr.cleanup-all.{uuid.uuid4().hex}"
    cleaned = []
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", EngineCache(maxsize=4))

    class Engine:
        def process_image(self, image, **kwargs):
            return []

        def cleanup(self):
            cleaned.append(self)

    register_ocr_engine(name, Engine)
    run_ocr(target=_Target(), engine_name=name, resolution=72, context=object(), device="cpu")
    run_ocr(target=_Target(), engine_name=name, resolution=72, context=object(), device="cpu")

    assert cleanup_engine() == 2
    assert len(cleaned) == 2


def test_builtin_classic_path_still_initializes_and_reuses_engine_cache(monkeypatch):
    name = f"ocr.builtin-cache.{uuid.uuid4().hex}"
    cache = EngineCache(maxsize=4)
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", cache)
    initialized = []

    class BuiltinLikeEngine:
        def __init__(self, **kwargs):
            self._initialized = False

        def _initialize_model(self, languages, device, options):
            initialized.append((self, languages, device, options))

        def process_image(self, image, **kwargs):
            return []

    register_engine(name, EngineEntry(engine_type="classic", provider=BuiltinLikeEngine))

    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=object(),
        languages=["en"],
        device="cpu",
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=object(),
        languages=["en"],
        device="cpu",
    )
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=object(),
        languages=["fr"],
        device="cpu",
    )

    assert len(initialized) == 2
    assert initialized[0][1:] == (["en"], "cpu", None)
    assert initialized[1][1:] == (["fr"], "cpu", None)
