from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Type, Union

from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from natural_pdf.analyzers.guides.base import Guides, GuidesContext

_GUIDES_CLASS: Optional[Type["Guides"]] = None


def _load_guides_class() -> Type["Guides"]:
    global _GUIDES_CLASS
    if _GUIDES_CLASS is None:
        from natural_pdf.analyzers.guides.base import Guides

        _GUIDES_CLASS = Guides
    return _GUIDES_CLASS


class GuidesService:
    """Factory helpers that build Guides objects for host contexts."""

    def __init__(self, context):
        self._context = context

    @register_delegate("guides", "guides")
    def guides(
        self,
        host,
        verticals: Optional[Union[Iterable[float], "GuidesContext"]] = None,
        horizontals: Optional[Iterable[float]] = None,
        *,
        context=None,
        **kwargs,
    ) -> "Guides":
        Guides = _load_guides_class()
        effective_context = context if context is not None else host
        return Guides(
            verticals=verticals,
            horizontals=horizontals,
            context=effective_context,
            **kwargs,
        )
