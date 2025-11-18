"""Vision helper functions delegating to the shared VisualSearchService."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from natural_pdf.services.base import resolve_service

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region
    from natural_pdf.vision.results import MatchResults

ExampleInput = Union["Element", "Region"]
ExampleSequence = Sequence[ExampleInput]


def match_template(
    self,
    examples: Union[ExampleInput, ExampleSequence],
    confidence: float = 0.6,
    sizes: Optional[Union[float, Tuple, List]] = (0.8, 1.2),
    resolution: int = 72,
    hash_size: int = 20,
    step: Optional[int] = None,
    method: str = "phash",
    max_per_page: Optional[int] = None,
    show_progress: bool = True,
    mask_threshold: Optional[float] = None,
) -> "MatchResults":
    """Run visual template matching through the vision service."""

    service = resolve_service(self, "vision")
    return service.match_template(
        self,
        examples=examples,
        confidence=confidence,
        sizes=sizes,
        resolution=resolution,
        hash_size=hash_size,
        step=step,
        method=method,
        max_per_page=max_per_page,
        show_progress=show_progress,
        mask_threshold=mask_threshold,
    )


def find_similar(
    self,
    examples: Union[ExampleInput, ExampleSequence],
    using: str = "vision",
    confidence: float = 0.6,
    sizes: Optional[Union[float, Tuple, List]] = (0.8, 1.2),
    resolution: int = 72,
    hash_size: int = 20,
    step: Optional[int] = None,
    method: str = "phash",
    max_per_page: Optional[int] = None,
    show_progress: bool = True,
    mask_threshold: Optional[float] = None,
) -> "MatchResults":
    """Backward-compatible wrapper for the deprecated find_similar API."""

    service = resolve_service(self, "vision")
    return service.find_similar(
        self,
        examples=examples,
        using=using,
        confidence=confidence,
        sizes=sizes,
        resolution=resolution,
        hash_size=hash_size,
        step=step,
        method=method,
        max_per_page=max_per_page,
        show_progress=show_progress,
        mask_threshold=mask_threshold,
    )


__all__ = ["match_template", "find_similar"]
