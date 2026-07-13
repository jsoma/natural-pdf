"""Task-local state used by directional navigation.

The public ``Region.within()`` API is a temporary constraint.  It must not
mutate process-wide options: callers may navigate different PDFs concurrently,
and nested contexts must restore the outer constraint reliably.  A
``ContextVar`` supplies exactly those task- and thread-local semantics.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.region import Region


_directional_within: ContextVar[Optional["Region"]] = ContextVar(
    "natural_pdf_directional_within", default=None
)


def get_directional_within() -> Optional["Region"]:
    """Return the active task-local ``Region.within()`` constraint, if any."""

    return _directional_within.get()


def set_directional_within(region: "Region") -> Token[Optional["Region"]]:
    """Activate *region* as the task-local directional constraint.

    The returned token must be passed to :func:`reset_directional_within`.
    Keeping reset explicit makes nested contexts and exception restoration
    deterministic.
    """

    return _directional_within.set(region)


def reset_directional_within(token: Token[Optional["Region"]]) -> None:
    """Restore the directional constraint that preceded *token*."""

    _directional_within.reset(token)
