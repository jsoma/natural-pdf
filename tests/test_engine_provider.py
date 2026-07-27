"""EngineProvider and OCR integration smoke tests."""

from __future__ import annotations

import gc
import threading
import weakref

import pytest
from PIL import Image

import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.engine_registry import register_builtin
from natural_pdf.ocr.unified_dispatch import run_ocr


def test_engine_provider_caches_instances() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True  # Skip entry point discovery for the test

    call_count = {"value": 0}

    def factory(*, context=None, **opts):
        call_count["value"] += 1
        return object()

    provider.register("demo", "alpha", factory)

    ctx = object()
    first = provider.get("demo", context=ctx, name="alpha")
    second = provider.get("demo", context=ctx, name="alpha")

    assert first is second
    assert call_count["value"] == 1


def test_engine_provider_default_cache_isolated_by_context_and_nested_options() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    created = []

    def factory(*, context=None, **opts):
        engine = object()
        created.append((context, opts, engine))
        return engine

    provider.register("demo", "scoped", factory)
    first_context = object()
    second_context = object()

    first = provider.get(
        "demo",
        context=first_context,
        name="scoped",
        config={"languages": ["en", "fr"], "thresholds": {0.5, 0.8}},
    )
    equivalent = provider.get(
        "demo",
        context=first_context,
        name="scoped",
        config={"thresholds": {0.8, 0.5}, "languages": ["en", "fr"]},
    )
    different_options = provider.get(
        "demo",
        context=first_context,
        name="scoped",
        config={"languages": ["en"], "thresholds": {0.5, 0.8}},
    )
    different_context = provider.get(
        "demo",
        context=second_context,
        name="scoped",
        config={"languages": ["en", "fr"], "thresholds": {0.5, 0.8}},
    )

    assert equivalent is first
    assert different_options is not first
    assert different_context is not first
    assert len(created) == 3


def test_engine_provider_handles_cyclic_option_containers() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    provider.register("demo", "cyclic", lambda **_: object())
    context = []
    config = {}
    config["self"] = config

    first = provider.get("demo", context=context, name="cyclic", config=config)
    assert provider.get("demo", context=context, name="cyclic", config=config) is first


def test_engine_provider_keys_opaque_options_by_identity() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    provider.register("demo", "client", lambda **_: object())
    context = object()
    first_client = object()
    second_client = object()

    first = provider.get("demo", context=context, name="client", client=first_client)
    repeated = provider.get("demo", context=context, name="client", client=first_client)
    other = provider.get("demo", context=context, name="client", client=second_client)

    assert repeated is first
    assert other is not first


def test_engine_provider_retains_opaque_option_identity_until_eviction() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    provider.register("demo", "client", lambda **_: object())

    class Client:
        pass

    context = []  # non-weakrefable so it remains an active cache scope
    client = Client()
    client_ref = weakref.ref(client)
    provider.get("demo", context=context, name="client", client=client)
    del client
    gc.collect()

    assert client_ref() is not None
    assert provider.evict("demo", "client") == 1
    gc.collect()
    assert client_ref() is None


def test_engine_provider_evicts_when_weak_context_is_collected() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    class Engine:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    engine = Engine()
    provider.register("demo", "retained", lambda **_: engine)

    class Context:
        pass

    context = Context()
    context_ref = weakref.ref(context)
    provider.get("demo", context=context, name="retained")
    del context
    gc.collect()

    assert context_ref() is None
    assert engine.cleaned
    assert provider.evict("demo", "retained") == 0


def test_engine_provider_safely_retains_non_weakrefable_context_until_eviction() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    provider.register("demo", "retained", lambda **_: object())
    context = []

    first = provider.get("demo", context=context, name="retained")
    assert provider.get("demo", context=context, name="retained") is first
    assert provider.evict("demo", "retained") == 1


def test_engine_provider_explicit_singleton_and_custom_cache_key() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    provider.register("demo", "shared", lambda **_: object(), lifetime="singleton")
    provider.register(
        "demo",
        "tenant",
        lambda **_: object(),
        cache_key=lambda _context, options: options["tenant"],
    )

    assert provider.get("demo", context=object(), name="shared", ignored=1) is provider.get(
        "demo", context=object(), name="shared", ignored=2
    )
    assert provider.get("demo", context=object(), name="tenant", tenant="a") is provider.get(
        "demo", context=object(), name="tenant", tenant="a"
    )
    assert provider.get("demo", context=object(), name="tenant", tenant="a") is not provider.get(
        "demo", context=object(), name="tenant", tenant="b"
    )


def test_builtin_registration_explicitly_preserves_singleton_sharing() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    register_builtin(provider, "demo", "builtin", lambda **_: object())

    assert provider.get("demo", context=object(), name="builtin") is provider.get(
        "demo", context=object(), name="builtin"
    )


def test_engine_provider_rejects_invalid_lifetime_and_cache_keys() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    with pytest.raises(ValueError, match="lifetime"):
        provider.register("demo", "bad", lambda **_: object(), lifetime="forever")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="context-lifetime"):
        provider.register(
            "demo",
            "bad",
            lambda **_: object(),
            lifetime="singleton",
            cache_key=lambda _context, _options: "one",
        )

    provider.register(
        "demo",
        "bad-key",
        lambda **_: object(),
        cache_key=lambda _context, _options: [],  # type: ignore[return-value]
    )
    with pytest.raises(TypeError, match="hashable"):
        provider.get("demo", context=object(), name="bad-key")


def test_engine_provider_transient_instances_are_caller_owned() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    class Engine:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    provider.register("demo", "temporary", lambda **_: Engine(), lifetime="transient")
    first = provider.get("demo", context=None, name="temporary")
    second = provider.get("demo", context=None, name="temporary")

    assert first is not second
    assert provider.evict("demo", "temporary") == 0
    assert not first.cleaned and not second.cleaned


def test_engine_provider_checkout_cleans_transient_exactly_once() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    cleanups = []

    class Engine:
        def cleanup(self):
            cleanups.append(self)

    provider.register("demo", "temporary", lambda **_: Engine(), lifetime="transient")

    with provider.checkout("demo", context=None, name="temporary") as engine:
        assert isinstance(engine, Engine)
        assert cleanups == []
    assert cleanups == [engine]

    # A second checkout creates and cleans a second, distinct instance.
    with provider.checkout("demo", context=None, name="temporary") as second:
        pass
    assert second is not engine
    assert cleanups == [engine, second]


def test_engine_provider_checkout_cleans_transient_on_error_and_uses_close() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    closed = []

    class Engine:
        def close(self):
            closed.append(self)

    provider.register("demo", "temporary", lambda **_: Engine(), lifetime="transient")

    with pytest.raises(RuntimeError, match="boom"):
        with provider.checkout("demo", context=None, name="temporary"):
            raise RuntimeError("boom")
    assert len(closed) == 1


def test_engine_provider_checkout_never_cleans_cached_lifetimes() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    cleanups = []

    class Engine:
        def cleanup(self):
            cleanups.append(self)

    provider.register("demo", "shared", lambda **_: Engine(), lifetime="singleton")
    provider.register("demo", "scoped", lambda **_: Engine(), lifetime="context")

    ctx = object()
    with provider.checkout("demo", context=ctx, name="shared") as singleton_engine:
        pass
    with provider.checkout("demo", context=ctx, name="scoped") as context_engine:
        pass

    assert cleanups == []
    # Cached instances are still owned (and served) by the provider.
    assert provider.get("demo", context=ctx, name="shared") is singleton_engine
    assert provider.get("demo", context=ctx, name="scoped") is context_engine


def test_engine_provider_evicts_before_cleanup_and_survives_cleanup_errors() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    cleanup_observations = []

    class Engine:
        def __init__(self, fail=False):
            self.fail = fail

        def cleanup(self):
            cleanup_observations.append(provider.clear("demo"))
            if self.fail:
                raise RuntimeError("cleanup failed")

    provider.register("demo", "first", lambda **_: Engine(fail=True))
    provider.register("demo", "second", lambda **_: Engine())
    context = object()
    provider.get("demo", context=context, name="first")
    provider.get("demo", context=context, name="second")

    assert provider.clear("demo") == 2
    assert cleanup_observations == [0, 0]


def test_engine_provider_replacement_is_visible_before_old_cleanup() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    replacement = object()
    observed = []
    cleanup_lock_free = []

    class OldEngine:
        def cleanup(self):
            completed = threading.Event()

            def inspect_provider():
                provider.list("demo")
                completed.set()

            observer = threading.Thread(target=inspect_provider)
            observer.start()
            observer.join(timeout=1)
            cleanup_lock_free.append(completed.is_set())
            observed.append(provider.get("demo", context=None, name="replaceable"))

    provider.register("demo", "replaceable", lambda **_: OldEngine(), lifetime="singleton")
    old = provider.get("demo", context=None, name="replaceable")
    provider.register(
        "demo",
        "replaceable",
        lambda **_: replacement,
        lifetime="singleton",
        replace=True,
    )

    assert observed == [replacement]
    assert observed[0] is not old
    assert cleanup_lock_free == [True]


def test_engine_provider_uses_close_when_cleanup_is_unavailable() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    class Engine:
        closed = False

        def close(self):
            self.closed = True

    engine = Engine()
    provider.register("demo", "closeable", lambda **_: engine, lifetime="singleton")
    provider.get("demo", context=None, name="closeable")

    assert provider.evict("demo", "closeable") == 1
    assert engine.closed


def test_engine_provider_does_not_clean_shared_object_until_last_reference_is_evicted() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True

    class Engine:
        cleanup_count = 0

        def cleanup(self):
            self.cleanup_count += 1

    engine = Engine()
    provider.register("first", "shared", lambda **_: engine, lifetime="singleton")
    provider.register("second", "shared", lambda **_: engine, lifetime="singleton")
    provider.get("first", context=None, name="shared")
    provider.get("second", context=None, name="shared")

    assert provider.evict("first", "shared") == 1
    assert engine.cleanup_count == 0
    assert provider.evict("second", "shared") == 1
    assert engine.cleanup_count == 1


def test_engine_provider_serializes_concurrent_creation_and_eviction() -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    creation_count = 0
    creation_lock = threading.Lock()

    def factory(**_):
        nonlocal creation_count
        with creation_lock:
            creation_count += 1
        return object()

    provider.register("demo", "concurrent", factory)
    context = object()
    results = []

    def get_engine():
        results.append(provider.get("demo", context=context, name="concurrent"))

    threads = [threading.Thread(target=get_engine) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(engine) for engine in results}) == 1
    assert creation_count == 1
    assert provider.evict("demo", "concurrent") == 1
    assert provider.get("demo", context=context, name="concurrent") is not results[0]
    assert creation_count == 2


def test_run_ocr_engine_with_custom_provider(monkeypatch) -> None:
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _FakeOCREngine:
        def __init__(self):
            self.calls = []

        def process_image(self, image=None, **kwargs):
            self.calls.append(kwargs)
            return [{"bbox": [0, 0, 1, 1], "text": "hi", "confidence": 0.9}]

    fake_engine = _FakeOCREngine()

    def factory(*, context=None, **opts):
        return fake_engine

    provider.register("ocr", "test-ocr", factory, replace=True)

    class Target:
        def render(self, resolution=72, **kwargs):
            return Image.new("RGB", (4, 4), color="white")

    result = run_ocr(
        target=Target(),
        engine_name="test-ocr",
        resolution=72,
        context=object(),
        languages=["en"],
        min_confidence=0.1,
        device="cpu",
        detect_only=False,
        options=None,
    )

    assert fake_engine.calls, "Engine should have been invoked"
    assert result.results[0]["text"] == "hi"
