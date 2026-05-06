"""OCR registry helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Type

from .base import register_engine

if TYPE_CHECKING:
    from natural_pdf.ocr.ocr_options import BaseOCROptions

__all__ = ["register_ocr_engine"]


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
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a custom OCR engine.

    Classic engines provide a factory that returns an object with
    ``process_image(...)``. VLM engines register a shorthand that resolves to an
    existing VLM OCR family/parser.
    """

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("OCR engine name must be a non-empty string.")

    normalized_kind = kind.strip().lower()
    if normalized_kind in {"classic", "ocr"}:
        from natural_pdf.ocr.unified_dispatch import EngineEntry
        from natural_pdf.ocr.unified_dispatch import register_engine as register_unified_engine

        if factory is None:
            raise ValueError("Classic OCR registration requires a factory.")
        entry = EngineEntry(
            engine_type="classic",
            provider=factory,
            options_class=options_class,
            needs_gpu_lock=needs_gpu_lock,
            install_hint=install_hint,
        )
        register_unified_engine(normalized_name, entry)

        provider_metadata = dict(metadata or {})
        if install_hint:
            provider_metadata.setdefault("install_hint", install_hint)
        provider_metadata.setdefault("kind", "classic")
        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            register_engine(
                capability,
                normalized_name,
                factory,
                replace=replace,
                metadata=provider_metadata,
            )
        return

    if normalized_kind in {"vlm", "vlm_shorthand"}:
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
        )
        register_unified_engine(normalized_name, entry)

        return

    raise ValueError("kind must be 'classic' or 'vlm'.")
