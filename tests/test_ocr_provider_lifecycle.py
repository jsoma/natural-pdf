"""Regressions for OCR dispatch through EngineProvider lifecycle ownership."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

import pytest
from PIL import Image

import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.engine_registry import register_ocr_engine
from natural_pdf.ocr.ocr_options import BaseOCROptions
from natural_pdf.ocr.ocr_provider import cleanup_engine
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

    # Separately normalized options with the same semantic init key reuse the
    # heavy model variant.
    run_ocr(
        target=_Target(),
        engine_name=name,
        resolution=72,
        context=first_context,
        languages=["en"],
        device="cpu",
        options=_CustomOptions(profile="accurate"),
    )
    assert len(constructed) == 4


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
    assert len(constructed) == 2


@pytest.mark.parametrize("availability_failure", ["false", "raise"])
def test_unavailable_provider_only_variant_is_evicted_and_retried(
    isolated_provider,
    availability_failure,
):
    name = f"ocr.provider-unavailable.{uuid.uuid4().hex}"
    good_context = object()
    bad_context = object()
    constructed = []
    cleaned = []

    class Engine:
        def __init__(self, context):
            self.context = context
            constructed.append(self)

        def is_available(self):
            if self.context is bad_context:
                if availability_failure == "raise":
                    raise RuntimeError("availability probe failed")
                return False
            return True

        def process_image(self, image, **kwargs):
            return []

        def cleanup(self):
            cleaned.append(self)

    isolated_provider.register(
        "ocr.apply",
        name,
        lambda *, context: Engine(context),
    )

    run_ocr(target=_Target(), engine_name=name, resolution=72, context=good_context)
    good_engine = constructed[0]
    for _ in range(2):
        error = "availability probe failed" if availability_failure == "raise" else "not available"
        with pytest.raises(RuntimeError, match=error):
            run_ocr(target=_Target(), engine_name=name, resolution=72, context=bad_context)

    # The valid context variant remains cached, while every failed variant is
    # detached, cleaned after checkout exit, and freshly reconstructed.
    run_ocr(target=_Target(), engine_name=name, resolution=72, context=good_context)
    assert len(constructed) == 3
    assert constructed[0] is good_engine
    assert cleaned == constructed[1:]


def test_uncacheable_custom_options_create_and_clean_one_engine_per_run(isolated_provider):
    name = f"ocr.uncacheable.{uuid.uuid4().hex}"
    constructed = []
    cleaned = []

    class MutableCallback:
        pass

    class Engine:
        def __init__(self):
            constructed.append(self)

        def process_image(self, image, **kwargs):
            return []

        def cleanup(self):
            cleaned.append(self)

    register_ocr_engine(name, Engine, options_class=_CustomOptions)
    context = object()
    option_mapping = {"extra_args": {"callback": MutableCallback()}}

    for _ in range(2):
        run_ocr(
            target=_Target(),
            engine_name=name,
            resolution=72,
            context=context,
            options=option_mapping,
            device="cpu",
        )

    assert len(constructed) == 2
    assert cleaned == constructed
    assert isolated_provider.evict("ocr.apply", name) == 0


def test_transient_custom_ocr_engine_is_cleaned_after_processing(isolated_provider):
    name = f"ocr.transient.{uuid.uuid4().hex}"
    events = []

    class Engine:
        def process_image(self, image, **kwargs):
            events.append("process")
            return []

        def cleanup(self):
            events.append("cleanup")

    register_ocr_engine(name, Engine, lifetime="transient")
    run_ocr(target=_Target(), engine_name=name, resolution=72, context=object())

    assert events == ["process", "cleanup"]


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
    cleaned = []

    class Engine:
        def is_available(self):
            return False

        def _initialize_model(self, languages, device, options):  # pragma: no cover
            raise AssertionError("unavailable engines must not initialize")

        def process_image(self, image, **kwargs):  # pragma: no cover
            raise AssertionError("unavailable engines must not run")

        def cleanup(self):
            cleaned.append(self)

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
    assert cleaned == constructed


def test_provider_capability_paths_forward_options(isolated_provider):
    apply_name = f"ocr.provider-apply.{uuid.uuid4().hex}"
    fallback_name = f"ocr.provider-fallback.{uuid.uuid4().hex}"
    received = []

    class ProviderOptions(BaseOCROptions):
        pass

    class Engine:
        def process_image(self, image=None, images=None, **kwargs):
            return []

    def factory(*, context, options):
        received.append((context, options))
        return Engine()

    # "ocr.apply" is the primary provider capability; bare "ocr" is the fallback
    # path when no "ocr.apply" registration exists (see _run_via_provider).
    isolated_provider.register("ocr.apply", apply_name, factory)
    isolated_provider.register("ocr", fallback_name, factory)
    apply_context = object()
    fallback_context = object()
    apply_options = ProviderOptions()
    fallback_options = ProviderOptions()

    run_ocr(
        target=_Target(),
        context=apply_context,
        engine_name=apply_name,
        resolution=72,
        options=apply_options,
    )
    run_ocr(
        target=_Target(),
        context=fallback_context,
        engine_name=fallback_name,
        resolution=72,
        options=fallback_options,
    )

    assert received == [(apply_context, apply_options), (fallback_context, fallback_options)]


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


def test_inflight_old_builtin_factory_cannot_publish_after_registration_change(
    isolated_provider,
    monkeypatch,
):
    name = f"ocr.stale-factory.{uuid.uuid4().hex}"
    cache = EngineCache(maxsize=4)
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", cache)
    factory_started = threading.Event()
    release_factory = threading.Event()
    created = []
    cleaned = []
    results = []
    errors = []

    class Engine:
        def __init__(self, label):
            self.label = label
            created.append(self)

        def process_image(self, image, **kwargs):
            return [{"bbox": [0, 0, 1, 1], "text": self.label, "confidence": 1.0}]

        def cleanup(self):
            cleaned.append(self)

    def slow_factory():
        engine = Engine("old")
        factory_started.set()
        if not release_factory.wait(timeout=2):  # pragma: no cover - deadlock diagnostic
            raise RuntimeError("factory release timed out")
        return engine

    old_entry = EngineEntry(
        engine_type="classic",
        provider=slow_factory,
        needs_gpu_lock=False,
    )
    replacement_entry = EngineEntry(
        engine_type="classic",
        provider=lambda: Engine("replacement"),
        needs_gpu_lock=False,
    )
    register_engine(name, old_entry)

    def run_old_registration():
        try:
            result = run_ocr(target=_Target(), engine_name=name, resolution=72)
            results.append(result.results[0]["text"])
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    worker = threading.Thread(target=run_old_registration)
    worker.start()
    assert factory_started.wait(timeout=2)
    register_engine(name, replacement_entry)
    release_factory.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert results == ["old"]
    assert cache.invalidate(name) == 0
    assert cleaned == created


def test_builtin_inference_lease_defers_replacement_cleanup(
    isolated_provider,
    monkeypatch,
):
    name = f"ocr.active-inference.{uuid.uuid4().hex}"
    cache = EngineCache(maxsize=4)
    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", cache)
    inference_started = threading.Event()
    release_inference = threading.Event()
    cleaned = []
    results = []
    errors = []

    class Engine:
        def __init__(self, label, block=False):
            self.label = label
            self.block = block

        def process_image(self, image, **kwargs):
            if self.block:
                inference_started.set()
                if not release_inference.wait(timeout=2):  # pragma: no cover
                    raise RuntimeError("inference release timed out")
            return [{"bbox": [0, 0, 1, 1], "text": self.label, "confidence": 1.0}]

        def cleanup(self):
            cleaned.append(self)

    old_engine = Engine("old", block=True)
    replacement_engine = Engine("replacement")
    register_engine(
        name,
        EngineEntry(
            engine_type="classic",
            provider=lambda: old_engine,
            needs_gpu_lock=False,
        ),
    )

    def run_old_inference():
        try:
            result = run_ocr(target=_Target(), engine_name=name, resolution=72)
            results.append(result.results[0]["text"])
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    worker = threading.Thread(target=run_old_inference)
    worker.start()
    assert inference_started.wait(timeout=2)
    register_engine(
        name,
        EngineEntry(
            engine_type="classic",
            provider=lambda: replacement_engine,
            needs_gpu_lock=False,
        ),
    )
    assert cleaned == []

    release_inference.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
    assert results == ["old"]
    assert cleaned == [old_engine]

    replacement = run_ocr(target=_Target(), engine_name=name, resolution=72)
    assert replacement.results[0]["text"] == "replacement"
    assert cache.clear() == 1
    assert cleaned == [old_engine, replacement_engine]
