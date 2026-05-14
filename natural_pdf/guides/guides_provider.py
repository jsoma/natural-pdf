"""Provider utilities for guide detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_guides_engine

logger = logging.getLogger(__name__)

Axis = Literal["vertical", "horizontal"]


@dataclass
class GuidesDetectionResult:
    coordinates: Sequence[float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GuidesBothDetectionResult:
    vertical: Sequence[float]
    horizontal: Sequence[float]
    metadata: Optional[Dict[str, Any]] = None


class GuidesEngine:
    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: Any,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        raise NotImplementedError

    def detect_both(
        self,
        *,
        method: str,
        context: Any,
        options: Dict[str, Any],
    ) -> GuidesBothDetectionResult:
        """Detect both guide axes. Engines can override to share expensive setup."""
        vertical_options = dict(options)
        horizontal_options = dict(options)
        if "max_lines_h" in vertical_options:
            vertical_options["max_lines_h"] = None
        if "max_lines_v" in horizontal_options:
            horizontal_options["max_lines_v"] = None

        vertical = self.detect(
            axis="vertical",
            method=method,
            context=context,
            options=vertical_options,
        )
        horizontal = self.detect(
            axis="horizontal",
            method=method,
            context=context,
            options=horizontal_options,
        )
        metadata: Dict[str, Any] = {}
        if vertical.metadata:
            metadata["vertical"] = vertical.metadata
        if horizontal.metadata:
            metadata["horizontal"] = horizontal.metadata
        return GuidesBothDetectionResult(
            vertical=vertical.coordinates,
            horizontal=horizontal.coordinates,
            metadata=metadata or None,
        )


DEFAULT_GUIDE_ENGINES = {
    "content": "builtin.content",
    "lines": "builtin.lines",
    "whitespace": "builtin.whitespace",
    "headers": "builtin.headers",
    "stripes": "builtin.stripes",
}


class _RouterGuidesEngine(GuidesEngine):
    """Fallback engine that routes to method-specific builtins."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: Any,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        target_name = DEFAULT_GUIDE_ENGINES.get(method)
        if not target_name:
            raise LookupError(f"No built-in engine registered for guides method '{method}'.")
        provider = get_provider()
        engine = provider.get("guides.detect", context=context, name=target_name)
        return engine.detect(axis=axis, method=method, context=context, options=options)

    def detect_both(
        self,
        *,
        method: str,
        context: Any,
        options: Dict[str, Any],
    ) -> GuidesBothDetectionResult:
        target_name = DEFAULT_GUIDE_ENGINES.get(method)
        if not target_name:
            raise LookupError(f"No built-in engine registered for guides method '{method}'.")
        provider = get_provider()
        engine = provider.get("guides.detect", context=context, name=target_name)
        return engine.detect_both(method=method, context=context, options=options)


def register_guides_engines(provider=None) -> None:
    from natural_pdf.guides.engines.content import ContentGuidesEngine
    from natural_pdf.guides.engines.headers import HeadersGuidesEngine
    from natural_pdf.guides.engines.lines import LinesGuidesEngine
    from natural_pdf.guides.engines.stripes import StripesGuidesEngine
    from natural_pdf.guides.engines.whitespace import WhitespaceGuidesEngine

    def _register(name: str, factory: Callable[..., Any]) -> None:
        register_builtin(provider, "guides.detect", name, factory)

    _register("builtin.content", lambda **_: ContentGuidesEngine())
    _register("builtin.lines", lambda **_: LinesGuidesEngine())
    _register("builtin.whitespace", lambda **_: WhitespaceGuidesEngine())
    _register("builtin.headers", lambda **_: HeadersGuidesEngine())
    _register("builtin.stripes", lambda **_: StripesGuidesEngine())
    _register("builtin", lambda **_: _RouterGuidesEngine())


def run_guides_detect(
    *,
    axis: Axis,
    method: str,
    context: Any,
    options: Optional[Dict[str, Any]] = None,
    engine_name: Optional[str] = None,
) -> GuidesDetectionResult:
    provider = get_provider()
    resolved_name = engine_name or DEFAULT_GUIDE_ENGINES.get(method) or "builtin"
    if resolved_name is None:
        raise ValueError(f"No default engine registered for guides method '{method}'.")
    try:
        engine = provider.get("guides.detect", context=context, name=resolved_name)
    except LookupError as exc:
        if resolved_name != "builtin":
            engine = provider.get("guides.detect", context=context, name="builtin")
        else:
            raise exc
    return engine.detect(axis=axis, method=method, context=context, options=options or {})


def run_guides_detect_both(
    *,
    method: str,
    context: Any,
    options: Optional[Dict[str, Any]] = None,
    engine_name: Optional[str] = None,
) -> GuidesBothDetectionResult:
    provider = get_provider()
    resolved_name = engine_name or DEFAULT_GUIDE_ENGINES.get(method) or "builtin"
    if resolved_name is None:
        raise ValueError(f"No default engine registered for guides method '{method}'.")
    try:
        engine = provider.get("guides.detect", context=context, name=resolved_name)
    except LookupError as exc:
        if resolved_name != "builtin":
            engine = provider.get("guides.detect", context=context, name="builtin")
        else:
            raise exc
    return engine.detect_both(method=method, context=context, options=options or {})


register_guides_engines()


__all__ = [
    "GuidesBothDetectionResult",
    "GuidesDetectionResult",
    "run_guides_detect",
    "run_guides_detect_both",
    "register_guides_engines",
]
