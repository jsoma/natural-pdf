from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TextMixin:  # pylint: disable=too-few-public-methods
    """Mixin that adds general text-replacement capabilities.

    Two public entry points are exposed to any class that inherits this mix-in:

    1. ``update_text`` (preferred) –  iterate over text elements selected via the
       ``selector`` argument (default: ``"text"``) and apply a *correction* callback
       which optionally returns replacement text.  If the callback returns a
       non-``None`` string that differs from the current value, the element's
       ``text`` attribute is updated in-place.

    2. ``correct_ocr`` – legacy name kept for backward compatibility.  It simply
       forwards to :py:meth:`update_text` while forcing
       ``selector="text[source=ocr]"`` so that the historic behaviour (acting only
       on OCR-generated elements) is preserved.
    """

    # ---------------------------------------------------------------------
    # Back-compat shim
    # ---------------------------------------------------------------------
    def correct_ocr(self, *args, selector: str = "text[source=ocr]", **kwargs):  # type: ignore[override]
        """Backward-compatibility wrapper that forwards to *update_text*.

        Parameters
        ----------
        *args, **kwargs
            Forwarded verbatim to :py:meth:`update_text` (after injecting the
            ``selector`` default shown above).
        """

        # Delegate – subclasses may have overridden *update_text* with a richer
        # signature so we pass everything through untouched.
        return self.update_text(*args, selector=selector, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Generic fallback implementation
    # ------------------------------------------------------------------
    def update_text(  # type: ignore[override]
        self,
        transform: Callable[[Any], Optional[str]],
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
        **_,
    ):
        """Generic implementation that works for any object exposing *find_all*.

        Classes that require more sophisticated behaviour (parallelism, page
        delegation, etc.) are expected to *override* this method while keeping
        the same public contract.
        """

        if not callable(transform):
            raise TypeError("transform must be callable")

        # We rely on the presence of *find_all* to obtain elements.  If the
        # subclass does not implement it then it *must* override update_text.
        if not hasattr(self, "find_all"):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement `update_text` explicitly "
                "(no `find_all` method found)."
            )

        try:
            elements_collection = self.find_all(
                selector=selector, apply_exclusions=apply_exclusions
            )
        except Exception as exc:  # pragma: no cover – defensive
            raise RuntimeError(
                f"Failed to gather elements with selector '{selector}': {exc}"
            ) from exc

        # `find_all` returns an ElementCollection; fall back gracefully otherwise.
        elements_iter = getattr(elements_collection, "elements", elements_collection)
        updated = 0

        for element in elements_iter:
            if not hasattr(element, "text"):
                continue

            new_text = transform(element)
            if new_text is not None and isinstance(new_text, str) and new_text != element.text:
                element.text = new_text
                updated += 1

        logger.info(
            "%s.update_text – processed %d element(s); updated %d.",
            self.__class__.__name__,
            len(elements_iter),
            updated,
        )

        return self
