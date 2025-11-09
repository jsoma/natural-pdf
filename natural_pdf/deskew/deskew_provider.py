"""Deskew provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_deskew_engine
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)

try:  # Optional dependency
    from deskew import determine_skew  # type: ignore[import]
except ImportError:  # pragma: no cover
    determine_skew = None


@dataclass
class DeskewApplyResult:
    image: Optional[Image.Image]
    angle: Optional[float]


def register_deskew_engines(provider=None) -> None:
    engine = _DefaultDeskewEngine()

    def factory(**_):
        return engine

    for capability in ("deskew", "deskew.detect", "deskew.apply"):
        register_builtin(provider, capability, "standard", factory)


def run_deskew_detect(
    *,
    target: Any,
    context: Any,
    engine_name: Optional[str] = None,
    resolution: int = 72,
    grayscale: bool = True,
    deskew_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    provider = get_provider()
    name = (engine_name or "standard").strip().lower()
    try:
        engine = provider.get("deskew.detect", context=context, name=name)
    except LookupError:
        engine = provider.get("deskew", context=context, name=name)
    return engine.detect(
        target=target,
        context=context,
        resolution=resolution,
        grayscale=grayscale,
        deskew_kwargs=deskew_kwargs or {},
    )


def run_deskew_apply(
    *,
    target: Any,
    context: Any,
    engine_name: Optional[str] = None,
    resolution: int = 300,
    angle: Optional[float] = None,
    detection_resolution: int = 72,
    grayscale: bool = True,
    deskew_kwargs: Optional[Dict[str, Any]] = None,
) -> DeskewApplyResult:
    provider = get_provider()
    name = (engine_name or "standard").strip().lower()
    try:
        engine = provider.get("deskew.apply", context=context, name=name)
    except LookupError:
        engine = provider.get("deskew", context=context, name=name)
    return engine.apply(
        target=target,
        context=context,
        resolution=resolution,
        angle=angle,
        detection_resolution=detection_resolution,
        grayscale=grayscale,
        deskew_kwargs=deskew_kwargs or {},
    )


class _DefaultDeskewEngine:
    def detect(
        self,
        *,
        target: Any,
        context: Any,
        resolution: int,
        grayscale: bool,
        deskew_kwargs: Dict[str, Any],
    ) -> Optional[float]:
        if determine_skew is None:
            raise ImportError(
                "Deskew library not found. Install with: pip install natural-pdf[deskew]"
            )
        image = _render_target(target, resolution=resolution, grayscale=grayscale)
        img_np: NDArray[Any] = np.array(image)
        if grayscale and img_np.ndim == 3 and img_np.shape[2] >= 3:
            gray_np: NDArray[np.uint8] = np.mean(img_np[:, :, :3], axis=2).astype(np.uint8)
        elif grayscale and img_np.ndim == 2:
            gray_np = cast(NDArray[np.uint8], img_np)
        else:
            gray_np = cast(NDArray[np.uint8], img_np)
        raw_angle = cast(Optional[float], determine_skew(cast(Any, gray_np), **deskew_kwargs))
        if raw_angle is None:
            return None
        return float(raw_angle)

    def apply(
        self,
        *,
        target: Any,
        context: Any,
        resolution: int,
        angle: Optional[float],
        detection_resolution: int,
        grayscale: bool,
        deskew_kwargs: Dict[str, Any],
    ) -> DeskewApplyResult:
        rotation_angle = angle
        if rotation_angle is None:
            rotation_angle = self.detect(
                target=target,
                context=context,
                resolution=detection_resolution,
                grayscale=grayscale,
                deskew_kwargs=deskew_kwargs,
            )
        image: Image.Image = _render_target(target, resolution=resolution, grayscale=False)
        if rotation_angle is None or abs(rotation_angle) <= 0.05:
            return DeskewApplyResult(image=image, angle=rotation_angle)
        fill = (255, 255, 255) if image.mode == "RGB" else 255
        rotated = image.rotate(
            rotation_angle,
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=fill,
        )
        return DeskewApplyResult(image=rotated, angle=rotation_angle)


def _render_target(target: Any, *, resolution: int, grayscale: bool) -> Image.Image:
    render_fn = getattr(target, "render", None)
    if not callable(render_fn):
        raise AttributeError("Target does not support rendering.")
    with pdf_render_lock:
        image = render_fn(resolution=resolution)
    if image is None:
        raise RuntimeError("Render call returned None for deskew operation.")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Render call returned unsupported type {type(image)!r}")
    if grayscale and image.mode not in ("L", "I"):
        return image.convert("L")
    return image


try:  # Register built-in engine
    register_deskew_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register deskew engines")


__all__ = [
    "DeskewApplyResult",
    "register_deskew_engines",
    "run_deskew_apply",
    "run_deskew_detect",
]
