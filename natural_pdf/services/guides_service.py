from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from natural_pdf.analyzers.guides.base import Guides, GuidesContext
from natural_pdf.services.registry import register_delegate


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
        from natural_pdf.analyzers.guides.base import Guides

        effective_context = context if context is not None else host
        return Guides(
            verticals=verticals,
            horizontals=horizontals,
            context=effective_context,
            **kwargs,
        )
