"""Validated request objects shared by public OCR-capable hosts.

This module is intentionally independent of Page, Region, Flow, and OCR service
implementations.  Public host mixins normalize their user-facing arguments
here, then services execute one explicit recognition, detection, or function
request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, TypeAlias, Union

from natural_pdf.ocr.replacement import OCRReplaceMode, normalize_ocr_replace_mode

CustomOCRCallable: TypeAlias = Callable[[Any], Optional[str]]
OCRLayoutMode: TypeAlias = Optional[Union[bool, str]]


@dataclass(frozen=True, slots=True)
class OCRRecognitionRequest:
    """A validated request to recognize and persist text."""

    engine: Optional[str] = None
    options: Optional[Any] = None
    languages: Optional[Tuple[str, ...]] = None
    min_confidence: Optional[float] = None
    device: Optional[str] = None
    resolution: Optional[int] = None
    apply_exclusions: bool = True
    replace: OCRReplaceMode = "ocr"
    use_cache: bool = True
    model: Optional[str] = None
    client: Optional[Any] = None
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    max_new_tokens: Optional[int] = None
    layout: OCRLayoutMode = None
    preserve_markup: bool = False
    mode: Literal["recognition"] = "recognition"

    def service_kwargs(self) -> dict[str, Any]:
        """Return arguments understood by :class:`OCRService`."""

        return {
            "engine": self.engine,
            "options": self.options,
            "languages": list(self.languages) if self.languages is not None else None,
            "min_confidence": self.min_confidence,
            "device": self.device,
            "resolution": self.resolution,
            "detect_only": False,
            "apply_exclusions": self.apply_exclusions,
            "replace": self.replace,
            "use_cache": self.use_cache,
            "model": self.model,
            "client": self.client,
            "prompt": self.prompt,
            "instructions": self.instructions,
            "max_new_tokens": self.max_new_tokens,
            "layout": self.layout,
            "preserve_markup": self.preserve_markup,
        }


@dataclass(frozen=True, slots=True)
class OCRDetectionRequest:
    """A validated request to refresh persistent text-detection geometry."""

    engine: Optional[str] = None
    options: Optional[Any] = None
    languages: Optional[Tuple[str, ...]] = None
    min_confidence: Optional[float] = None
    device: Optional[str] = None
    resolution: Optional[int] = None
    apply_exclusions: bool = True
    use_cache: bool = True
    model: Optional[str] = None
    client: Optional[Any] = None
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    max_new_tokens: Optional[int] = None
    layout: OCRLayoutMode = None
    preserve_markup: bool = False
    mode: Literal["detection"] = "detection"

    def service_kwargs(self) -> dict[str, Any]:
        """Return arguments understood by :class:`OCRService`."""

        return {
            "engine": self.engine,
            "options": self.options,
            "languages": list(self.languages) if self.languages is not None else None,
            "min_confidence": self.min_confidence,
            "device": self.device,
            "resolution": self.resolution,
            "detect_only": True,
            # Detection owns a separate refresh policy; recognition replacement
            # is deliberately not exposed for this request type.
            "replace": "ocr",
            "apply_exclusions": self.apply_exclusions,
            "use_cache": self.use_cache,
            "model": self.model,
            "client": self.client,
            "prompt": self.prompt,
            "instructions": self.instructions,
            "max_new_tokens": self.max_new_tokens,
            "layout": self.layout,
            "preserve_markup": self.preserve_markup,
        }


@dataclass(frozen=True, slots=True)
class OCRFunctionRequest:
    """A validated request to recognize text with a caller-provided function."""

    function: CustomOCRCallable
    source_label: str = "custom-ocr"
    confidence: Optional[float] = None
    replace: OCRReplaceMode = "ocr"
    mode: Literal["function"] = "function"


OCRRequest: TypeAlias = Union[OCRRecognitionRequest, OCRDetectionRequest, OCRFunctionRequest]


def _normalize_languages(languages: Optional[list[str]]) -> Optional[Tuple[str, ...]]:
    if languages is None:
        return None
    if isinstance(languages, (str, bytes)):
        raise TypeError("languages must be a list of language codes, not a string")
    if any(not isinstance(language, str) for language in languages):
        raise TypeError("languages must contain only strings")
    normalized = tuple(language.strip() for language in languages)
    if any(not language.strip() for language in normalized):
        raise ValueError("languages cannot contain empty values")
    return normalized


def _normalize_optional_string(value: Optional[str], *, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    # Prompts and instructions can contain significant surrounding whitespace;
    # validate here but let their owning resolver decide whether to normalize.
    return value


def _validate_positive_integer(value: Optional[int], *, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer or None")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _validate_confidence(confidence: Optional[float], *, name: str) -> Optional[float]:
    if confidence is None:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError(f"{name} must be a number between 0 and 1, or None")
    normalized = float(confidence)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def normalize_ocr_request(
    *,
    engine: Optional[str] = None,
    options: Optional[Any] = None,
    languages: Optional[list[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    resolution: Optional[int] = None,
    detect_only: bool = False,
    apply_exclusions: bool = True,
    replace: OCRReplaceMode = "ocr",
    use_cache: bool = True,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    layout: OCRLayoutMode = None,
    preserve_markup: bool = False,
    function: Optional[CustomOCRCallable] = None,
    source_label: str = "custom-ocr",
    confidence: Optional[float] = None,
) -> OCRRequest:
    """Normalize one public OCR call into a strict discriminated request."""

    replace_mode = normalize_ocr_replace_mode(replace)
    if not isinstance(detect_only, bool):
        raise TypeError("detect_only must be a bool")
    if not isinstance(apply_exclusions, bool):
        raise TypeError("apply_exclusions must be a bool")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache must be a bool")
    if not isinstance(preserve_markup, bool):
        raise TypeError("preserve_markup must be a bool")
    if layout is not None and not isinstance(layout, (bool, str)):
        raise TypeError("layout must be a bool, string, or None")
    if function is not None and not callable(function):
        raise TypeError("function must be callable or None")

    normalized_engine = _normalize_optional_string(engine, name="engine")
    normalized_device = _normalize_optional_string(device, name="device")
    normalized_model = _normalize_optional_string(model, name="model")
    normalized_prompt = _normalize_optional_string(prompt, name="prompt")
    normalized_instructions = _normalize_optional_string(instructions, name="instructions")
    normalized_resolution = _validate_positive_integer(resolution, name="resolution")
    normalized_max_new_tokens = _validate_positive_integer(max_new_tokens, name="max_new_tokens")

    if function is not None:
        incompatible: list[str] = []
        neutral_values = {
            "engine": normalized_engine,
            "options": options,
            "languages": languages,
            "min_confidence": min_confidence,
            "device": normalized_device,
            "resolution": normalized_resolution,
            "model": normalized_model,
            "client": client,
            "prompt": normalized_prompt,
            "instructions": normalized_instructions,
            "max_new_tokens": normalized_max_new_tokens,
            "layout": layout,
        }
        incompatible.extend(name for name, value in neutral_values.items() if value is not None)
        if detect_only:
            incompatible.append("detect_only")
        if not apply_exclusions:
            incompatible.append("apply_exclusions")
        if not use_cache:
            incompatible.append("use_cache")
        if preserve_markup:
            incompatible.append("preserve_markup")
        if incompatible:
            joined = ", ".join(sorted(incompatible))
            raise ValueError(f"function OCR cannot be combined with: {joined}")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError("source_label must be a non-empty string")
        return OCRFunctionRequest(
            function=function,
            source_label=source_label.strip(),
            confidence=_validate_confidence(confidence, name="confidence"),
            replace=replace_mode,
        )

    if source_label != "custom-ocr" or confidence is not None:
        raise ValueError("source_label and confidence require function=")

    normalized_languages = _normalize_languages(languages)
    normalized_min_confidence = _validate_confidence(
        min_confidence,
        name="min_confidence",
    )
    if detect_only:
        if replace_mode != "ocr":
            raise ValueError(
                "detect_only=True refreshes detection artifacts and cannot use "
                f"recognition replace={replace_mode!r}"
            )
        return OCRDetectionRequest(
            engine=normalized_engine,
            options=options,
            languages=normalized_languages,
            min_confidence=normalized_min_confidence,
            device=normalized_device,
            resolution=normalized_resolution,
            apply_exclusions=apply_exclusions,
            use_cache=use_cache,
            model=normalized_model,
            client=client,
            prompt=normalized_prompt,
            instructions=normalized_instructions,
            max_new_tokens=normalized_max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
        )
    return OCRRecognitionRequest(
        engine=normalized_engine,
        options=options,
        languages=normalized_languages,
        min_confidence=normalized_min_confidence,
        device=normalized_device,
        resolution=normalized_resolution,
        apply_exclusions=apply_exclusions,
        replace=replace_mode,
        use_cache=use_cache,
        model=normalized_model,
        client=client,
        prompt=normalized_prompt,
        instructions=normalized_instructions,
        max_new_tokens=normalized_max_new_tokens,
        layout=layout,
        preserve_markup=preserve_markup,
    )


__all__ = [
    "CustomOCRCallable",
    "OCRDetectionRequest",
    "OCRFunctionRequest",
    "OCRRecognitionRequest",
    "OCRRequest",
    "normalize_ocr_request",
]
