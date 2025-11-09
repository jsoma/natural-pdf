"""Selector engine provider integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_selector_engine
from natural_pdf.selectors.host_mixin import SupportsSelectorHost

NATIVE_SELECTOR_ENGINE = "native"


@dataclass
class SelectorOptions:
    selector: str
    parsed: Optional[Dict[str, Any]]
    regex: bool
    case: bool
    reading_order: bool
    near_threshold: Optional[float]
    text_tolerance: Optional[Dict[str, Any]]
    auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]]
    extra: Dict[str, Any]


@dataclass
class SelectorContext:
    """
    Lightweight context wrapper that exposes a consistent view of the selector host.

    Engines should rely on this object instead of reaching directly into Page/Region/Flow
    internals so alternate hosts (or future types) automatically gain parity.
    """

    host: SupportsSelectorHost

    def parse(self, selector: str) -> Dict[str, Any]:
        from natural_pdf.selectors.parser import parse_selector

        return parse_selector(selector)

    @property
    def page(self) -> Any:
        return self.host.selector_page()

    @property
    def region(self) -> Any:
        return self.host.selector_region()

    @property
    def flow(self) -> Any:
        return self.host.selector_flow()


@dataclass
class SelectorResult:
    elements: ElementCollection
    diagnostics: Optional[Dict[str, Any]] = None


class SelectorEngine(Protocol):
    def query(
        self,
        *,
        context: SelectorContext,
        selector: str,
        options: SelectorOptions,
    ) -> SelectorResult: ...


class NativeSelectorEngine:
    """Default selector engine that delegates to the existing implementation."""

    def query(
        self,
        *,
        context: SelectorContext,
        selector: str,
        options: SelectorOptions,
    ) -> SelectorResult:
        from natural_pdf.core.selector_utils import _run_native_selector

        elements = _run_native_selector(
            context.host,
            selector,
            text_tolerance=options.text_tolerance,
            auto_text_tolerance=options.auto_text_tolerance,
            regex=options.regex,
            case=options.case,
            reading_order=options.reading_order,
            near_threshold=options.near_threshold,
        )
        return SelectorResult(elements=elements)


def register_selector_engines(provider=None) -> None:
    def factory(**_):
        return NativeSelectorEngine()

    register_builtin(provider, "selectors", NATIVE_SELECTOR_ENGINE, factory)


def resolve_selector_engine_name(host: Any, requested: Optional[str]) -> Optional[str]:
    candidate = _normalize_name(requested)
    if candidate:
        return candidate

    getter = getattr(host, "get_config", None)
    config_value = None
    if callable(getter):
        try:
            config_value = getter("selector_engine", None, scope="region")
        except TypeError:
            config_value = getter("selector_engine", None)

    candidate = _normalize_name(config_value)
    if candidate:
        return candidate

    try:
        from natural_pdf import options as npdf_options

        candidate = _normalize_name(getattr(npdf_options.selectors, "engine", None))
        if candidate:
            return candidate
    except Exception:  # pragma: no cover - best effort fallback
        return None

    return None


def run_selector_engine(
    host: Any,
    selector: str,
    *,
    engine_name: str,
    text_tolerance: Optional[Dict[str, Any]] = None,
    auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
    regex: bool = False,
    case: bool = True,
    reading_order: bool = True,
    near_threshold: Optional[float] = None,
) -> ElementCollection:
    if not isinstance(host, SupportsSelectorHost):
        raise TypeError(
            f"Selector host of type '{type(host).__name__}' does not implement SupportsSelectorHost."
        )

    provider = get_provider()
    engine = provider.get("selectors", context=host, name=engine_name)
    context = SelectorContext(host=host)
    options = SelectorOptions(
        selector=selector,
        parsed=None,
        regex=regex,
        case=case,
        reading_order=reading_order,
        near_threshold=near_threshold,
        text_tolerance=text_tolerance,
        auto_text_tolerance=auto_text_tolerance,
        extra={},
    )
    result = engine.query(context=context, selector=selector, options=options)
    return result.elements


def _normalize_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


try:  # Register on import
    register_selector_engines()
except Exception:  # pragma: no cover
    pass
