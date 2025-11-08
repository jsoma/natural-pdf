"""Selector clause registry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

FilterEntry = Dict[str, Any]
HandlerResult = Optional[Union[FilterEntry, List[FilterEntry]]]


@dataclass
class ClauseEvalContext:
    """Runtime context passed to pseudo/attribute handlers."""

    selector_context: Any
    aggregates: Dict[str, Any]
    options: Dict[str, Any]


PseudoHandler = Callable[[Dict[str, Any], ClauseEvalContext], HandlerResult]
AttributeHandler = Callable[[Dict[str, Any], ClauseEvalContext], HandlerResult]


_PSEUDO_HANDLERS: Dict[str, PseudoHandler] = {}
_ATTRIBUTE_HANDLERS: Dict[str, AttributeHandler] = {}


def register_pseudo(name: str, handler: Optional[PseudoHandler] = None, *, replace: bool = False):
    def decorator(func: PseudoHandler):
        normalized = _normalize_name(name)
        if not replace and normalized in _PSEUDO_HANDLERS:
            raise ValueError(f"Pseudo-class '{normalized}' already registered")
        _PSEUDO_HANDLERS[normalized] = func
        return func

    if handler is None:
        return decorator
    return decorator(handler)


def unregister_pseudo(name: str) -> None:
    _PSEUDO_HANDLERS.pop(_normalize_name(name), None)


def get_pseudo_handler(name: str) -> Optional[PseudoHandler]:
    return _PSEUDO_HANDLERS.get(_normalize_name(name))


def register_attribute(
    name: str, handler: Optional[AttributeHandler] = None, *, replace: bool = False
):
    def decorator(func: AttributeHandler):
        normalized = _normalize_name(name)
        if not replace and normalized in _ATTRIBUTE_HANDLERS:
            raise ValueError(f"Attribute handler '{normalized}' already registered")
        _ATTRIBUTE_HANDLERS[normalized] = func
        return func

    if handler is None:
        return decorator
    return decorator(handler)


def unregister_attribute(name: str) -> None:
    _ATTRIBUTE_HANDLERS.pop(_normalize_name(name), None)


def get_attribute_handler(name: str) -> Optional[AttributeHandler]:
    return _ATTRIBUTE_HANDLERS.get(_normalize_name(name))


def _normalize_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Handler name must be a non-empty string")
    return normalized
