from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable, List, Optional, Protocol, Sequence

from natural_pdf.services.base import resolve_service
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class SupportsFindAll(Protocol):
    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Sequence[str] | str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: Optional[dict] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
    ) -> Any: ...


class TextService:
    """Shared text update helpers formerly provided by TextMixin."""

    def __init__(self, _context) -> None:  # context reserved for future use
        self._context = _context

    @register_delegate("text", "correct_ocr")
    def correct_ocr(
        self,
        host: SupportsFindAll,
        transform: Callable[[Any], Optional[str]],
        *,
        apply_exclusions: bool = False,
    ):
        return self.update_text(
            host,
            transform=transform,
            selector="text[source=ocr]",
            apply_exclusions=apply_exclusions,
        )

    @register_delegate("text", "update_ocr")
    def update_ocr(
        self,
        host: SupportsFindAll,
        transform: Callable[[Any], Optional[str]],
        *,
        apply_exclusions: bool = False,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        show_progress: bool = False,
    ):
        return self.update_text(
            host,
            transform=transform,
            selector="text[source=ocr]",
            apply_exclusions=apply_exclusions,
            max_workers=max_workers,
            progress_callback=progress_callback,
            show_progress=show_progress,
        )

    @register_delegate("text", "update_text")
    def update_text(
        self,
        host: SupportsFindAll,
        transform: Callable[[Any], Optional[str]],
        *,
        selector: str = "text",
        apply_exclusions: bool = False,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        show_progress: bool = False,
    ):
        if not callable(transform):
            raise TypeError("transform must be callable")

        finder = getattr(host, "find_all", None)
        if finder is None:
            raise NotImplementedError(
                f"{host.__class__.__name__} must implement `update_text` explicitly "
                "(no `find_all` method found)."
            )

        try:
            elements_collection = finder(selector=selector, apply_exclusions=apply_exclusions)
        except Exception as exc:  # pragma: no cover – defensive
            raise RuntimeError(
                f"Failed to gather elements with selector '{selector}': {exc}"
            ) from exc

        elements = getattr(elements_collection, "elements", elements_collection)
        elements = list(elements)
        if not elements:
            logger.info(
                "%s.update_text – no elements matched selector '%s'.",
                host.__class__.__name__,
                selector,
            )
            return host

        element_pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm

                element_pbar = tqdm(
                    total=len(elements),
                    desc=f"Updating text {getattr(host, 'number', '')}".strip(),
                    unit="element",
                    leave=False,
                )
            except Exception:
                element_pbar = None

        processed_count = 0
        updated_count = 0
        error_count = 0

        def _process(element):
            try:
                corrected = transform(element)
                if corrected is not None and not isinstance(corrected, str):
                    logger.warning(
                        "%s.update_text – callback returned non-string type %s; skipping.",
                        host.__class__.__name__,
                        type(corrected),
                    )
                    return element, None, None
                return element, corrected, None
            except Exception as exc:
                logger.error(
                    "%s.update_text – error applying callback to element %r: %s",
                    host.__class__.__name__,
                    getattr(element, "bbox", None),
                    exc,
                    exc_info=False,
                )
                return element, None, exc
            finally:
                if element_pbar:
                    element_pbar.update(1)
                if progress_callback:
                    try:
                        progress_callback()
                    except Exception as cb_exc:
                        logger.error(
                            "%s.update_text – progress callback failed: %s",
                            host.__class__.__name__,
                            cb_exc,
                            exc_info=False,
                        )

        def _apply_result(element, corrected_text, error):
            if error:
                return False, True
            if corrected_text is not None and corrected_text != getattr(element, "text", None):
                element.text = corrected_text
                return True, False
            return False, False

        if max_workers and max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_process, element): element for element in elements}
                for future in concurrent.futures.as_completed(future_map):
                    processed_count += 1
                    element, corrected_text, error = future.result()
                    updated, errored = _apply_result(element, corrected_text, error)
                    if updated:
                        updated_count += 1
                    if errored:
                        error_count += 1
        else:
            for element in elements:
                processed_count += 1
                element, corrected_text, error = _process(element)
                updated, errored = _apply_result(element, corrected_text, error)
                if updated:
                    updated_count += 1
                if errored:
                    error_count += 1

        if element_pbar:
            element_pbar.close()

        logger.info(
            "%s.update_text – processed %d/%d element(s); updated %d; errors %d.",
            host.__class__.__name__,
            processed_count,
            len(elements),
            updated_count,
            error_count,
        )
        return host
