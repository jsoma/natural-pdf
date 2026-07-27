"""OCR registry helpers."""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Optional, Type

from natural_pdf.engine_provider import EngineCacheKey, EngineLifetime

from .base import register_engine

if TYPE_CHECKING:
    from natural_pdf.ocr.ocr_options import BaseOCROptions

__all__ = ["register_ocr_engine"]


def _call_custom_factory(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a public OCR factory with only the constructor inputs it accepts."""

    try:
        parameters = signature(factory).parameters
    except (TypeError, ValueError):
        return factory()

    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return factory(**kwargs)

    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    return factory(**accepted)


def _wrap_classic_factory(
    factory: Callable[..., Any],
    *,
    engine_name: str,
    install_hint: Optional[str],
) -> Callable[..., Any]:
    """Adapt the public factory API and initialize each provider variant."""

    def provider_factory(*, context: Any = None, **constructor_options: Any) -> Any:
        instance = _call_custom_factory(factory, context=context, **constructor_options)

        is_available = getattr(instance, "is_available", None)
        if callable(is_available) and not is_available():
            hint = install_hint or f"pip install {engine_name}"
            raise RuntimeError(
                f"OCR engine {engine_name!r} is not available. Install it with: {hint}"
            )

        initialize = getattr(instance, "_initialize_model", None)
        if callable(initialize):
            languages = list(constructor_options.get("languages") or ["en"])
            device = constructor_options.get("device") or "auto"
            initialize(languages, device, constructor_options.get("options"))
            if hasattr(instance, "_initialized"):
                instance._initialized = True
        return instance

    return provider_factory


def register_ocr_engine(
    name: str,
    factory: Optional[Callable[..., Any]] = None,
    *,
    kind: str = "classic",
    options_class: Optional[Type["BaseOCROptions"]] = None,
    install_hint: Optional[str] = None,
    model_resolver: Optional[Callable[[], str]] = None,
    vlm_family: Optional[str] = None,
    needs_gpu_lock: bool = True,
    cache_namespace: Optional[str] = None,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    lifetime: EngineLifetime = "context",
    cache_key: Optional[EngineCacheKey] = None,
) -> None:
    """Register a custom OCR engine.

    Classic engines provide a factory that returns an object with
    ``process_image(...)``. VLM engines register a shorthand that resolves to an
    existing VLM OCR family/parser.

    Persistent OCR result caching is disabled for custom engines unless
    ``cache_namespace`` is a non-empty, stable, non-secret identifier for the
    implementation and configuration (for example ``"acme-ocr:v2"``). Change
    the namespace whenever behavior that can affect OCR output changes. The
    namespace is hashed into cache keys and is never inferred from callable
    representations or object identities.

    ``lifetime`` and ``cache_key`` control EngineProvider instance caching for
    classic engines (see :func:`natural_pdf.engine_registry.base.register_engine`).
    VLM shorthands are not managed by the EngineProvider, so passing lifecycle
    controls with ``kind="vlm"`` raises ``ValueError``.
    """

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("OCR engine name must be a non-empty string.")

    normalized_kind = kind.strip().lower()
    if normalized_kind in {"classic", "ocr"}:
        from natural_pdf.engine_provider import get_provider
        from natural_pdf.ocr.unified_dispatch import EngineEntry, get_registry
        from natural_pdf.ocr.unified_dispatch import register_engine as register_unified_engine

        if factory is None:
            raise ValueError("Classic OCR registration requires a factory.")

        # Provider.register() deliberately skips existing registrations when
        # replace=False. Do the same before touching the unified metadata so
        # dispatch and provider resolution cannot disagree about the factory.
        if not replace:
            provider = get_provider()
            provider_conflict = any(
                normalized_name in provider.list(capability).get(capability, ())
                for capability in ("ocr", "ocr.apply", "ocr.extract")
            )
            if provider_conflict or normalized_name in get_registry():
                return

        provider_factory = _wrap_classic_factory(
            factory,
            engine_name=normalized_name,
            install_hint=install_hint,
        )
        entry = EngineEntry(
            # Custom classics are owned by EngineProvider. Built-ins retain
            # the specialized unified EngineCache used for model setup/LRU.
            engine_type="classic_provider",
            provider=provider_factory,
            provider_init_options=True,
            options_class=options_class,
            needs_gpu_lock=needs_gpu_lock,
            install_hint=install_hint,
            cache_namespace=cache_namespace,
        )

        provider_metadata = dict(metadata or {})
        if install_hint:
            provider_metadata.setdefault("install_hint", install_hint)
        provider_metadata.setdefault("kind", "classic")
        if cache_namespace:
            provider_metadata.setdefault("cache_namespace", cache_namespace)
        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            register_engine(
                capability,
                normalized_name,
                provider_factory,
                replace=replace,
                metadata=provider_metadata,
                lifetime=lifetime,
                cache_key=cache_key,
            )
        register_unified_engine(normalized_name, entry)
        return

    if normalized_kind in {"vlm", "vlm_shorthand"}:
        # VLM shorthands register only into unified dispatch — there is no
        # EngineProvider registration for lifecycle controls to apply to.
        if lifetime != "context" or cache_key is not None:
            raise ValueError(
                "lifetime/cache_key apply only to classic OCR engines; "
                "VLM shorthands are not managed by the EngineProvider."
            )
        from natural_pdf.ocr.unified_dispatch import EngineEntry
        from natural_pdf.ocr.unified_dispatch import register_engine as register_unified_engine

        if model_resolver is None:
            raise ValueError("VLM OCR registration requires model_resolver.")
        if not vlm_family:
            raise ValueError("VLM OCR registration requires vlm_family.")
        entry = EngineEntry(
            engine_type="vlm_shorthand",
            model_resolver=model_resolver,
            vlm_family=vlm_family,
            needs_gpu_lock=needs_gpu_lock,
            install_hint=install_hint,
            cache_namespace=cache_namespace,
        )
        register_unified_engine(normalized_name, entry)

        return

    raise ValueError("kind must be 'classic' or 'vlm'.")
