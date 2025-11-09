"""Selector clause registry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, overload


@dataclass
class ClauseEvalContext:
    """Runtime context passed to pseudo/attribute handlers."""

    selector_context: Any
    aggregates: Dict[str, Any]
    options: Dict[str, Any]


FilterEntry = Dict[str, Any]
HandlerResult = Optional[Union[FilterEntry, List[FilterEntry]]]
PostHandler = Callable[[List[Any], Dict[str, Any], ClauseEvalContext], List[Any]]
RelationalHandler = Callable[[List[Any], Dict[str, Any], ClauseEvalContext], List[Any]]


PseudoHandler = Callable[[Dict[str, Any], ClauseEvalContext], HandlerResult]
AttributeHandler = Callable[[Dict[str, Any], ClauseEvalContext], HandlerResult]

PseudoDecorator = Callable[[PseudoHandler], PseudoHandler]
AttributeDecorator = Callable[[AttributeHandler], AttributeHandler]
PostDecorator = Callable[[PostHandler], PostHandler]
RelationalDecorator = Callable[[RelationalHandler], RelationalHandler]


_PSEUDO_HANDLERS: Dict[str, PseudoHandler] = {}
_ATTRIBUTE_HANDLERS: Dict[str, AttributeHandler] = {}
_POST_HANDLERS: Dict[str, PostHandler] = {}
_RELATIONAL_HANDLERS: Dict[str, RelationalHandler] = {}


@overload
def register_pseudo(
    name: str, handler: PseudoHandler, *, replace: bool = False
) -> PseudoHandler: ...


@overload
def register_pseudo(
    name: str, handler: None = None, *, replace: bool = False
) -> PseudoDecorator: ...


def register_pseudo(
    name: str, handler: Optional[PseudoHandler] = None, *, replace: bool = False
) -> PseudoHandler | PseudoDecorator:
    def decorator(func: PseudoHandler) -> PseudoHandler:
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


@overload
def register_attribute(
    name: str, handler: AttributeHandler, *, replace: bool = False
) -> AttributeHandler: ...


@overload
def register_attribute(
    name: str, handler: None = None, *, replace: bool = False
) -> AttributeDecorator: ...


def register_attribute(
    name: str, handler: Optional[AttributeHandler] = None, *, replace: bool = False
) -> AttributeHandler | AttributeDecorator:
    def decorator(func: AttributeHandler) -> AttributeHandler:
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


@overload
def register_post_pseudo(
    name: str, handler: PostHandler, *, replace: bool = False
) -> PostHandler: ...


@overload
def register_post_pseudo(
    name: str, handler: None = None, *, replace: bool = False
) -> PostDecorator: ...


def register_post_pseudo(
    name: str, handler: Optional[PostHandler] = None, *, replace: bool = False
) -> PostHandler | PostDecorator:
    def decorator(func: PostHandler) -> PostHandler:
        normalized = _normalize_name(name)
        if not replace and normalized in _POST_HANDLERS:
            raise ValueError(f"Post-pseudo '{normalized}' already registered")
        _POST_HANDLERS[normalized] = func
        return func

    if handler is None:
        return decorator
    return decorator(handler)


def get_post_handler(name: str) -> Optional[PostHandler]:
    return _POST_HANDLERS.get(_normalize_name(name))


@overload
def register_relational_pseudo(
    name: str, handler: RelationalHandler, *, replace: bool = False
) -> RelationalHandler: ...


@overload
def register_relational_pseudo(
    name: str, handler: None = None, *, replace: bool = False
) -> RelationalDecorator: ...


def register_relational_pseudo(
    name: str, handler: Optional[RelationalHandler] = None, *, replace: bool = False
) -> RelationalHandler | RelationalDecorator:
    def decorator(func: RelationalHandler) -> RelationalHandler:
        normalized = _normalize_name(name)
        if not replace and normalized in _RELATIONAL_HANDLERS:
            raise ValueError(f"Relational pseudo '{normalized}' already registered")
        _RELATIONAL_HANDLERS[normalized] = func
        return func

    if handler is None:
        return decorator
    return decorator(handler)


def get_relational_handler(name: str) -> Optional[RelationalHandler]:
    return _RELATIONAL_HANDLERS.get(_normalize_name(name))


def _normalize_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Handler name must be a non-empty string")
    return normalized
