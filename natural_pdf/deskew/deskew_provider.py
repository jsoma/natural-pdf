"""Deskew provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)


@dataclass
class DeskewApplyResult:
    image: Optional[Image.Image]
    angle: Optional[float]


def register_deskew_engines(provider=None) -> None:
    projection = ProjectionProfileEngine()
    hough = HoughEngine()

    def proj_factory(**_):
        return projection

    def hough_factory(**_):
        return hough

    for capability in ("deskew", "deskew.detect", "deskew.apply"):
        register_builtin(provider, capability, "standard", proj_factory)
        register_builtin(provider, capability, "projection", proj_factory)
        register_builtin(provider, capability, "hough", hough_factory)


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


class ProjectionProfileEngine:
    """Projection-profile deskew engine (default).

    Uses coarse-to-fine rotation search maximizing row-sum variance.
    Dependencies: numpy, scipy (both core deps).

    Returns the **correction angle** in degrees — i.e. the value to pass to
    ``PIL.Image.rotate()`` to undo the detected skew.  For an image skewed
    +3° counter-clockwise the return value will be approximately -3°.
    """

    def detect(
        self,
        *,
        target: Any,
        context: Any,
        resolution: int,
        grayscale: bool,
        deskew_kwargs: Dict[str, Any],
    ) -> Optional[float]:
        from scipy.ndimage import rotate as ndi_rotate

        image = _render_target(target, resolution=resolution, grayscale=True)
        img_np: NDArray[np.uint8] = np.array(image)
        # Ensure 2-D even if render returned an RGB image
        if img_np.ndim == 3:
            img_np = img_np.mean(axis=2).astype(np.uint8)

        # Downsample for speed during coarse sweep
        max_dim = max(img_np.shape[:2])
        if max_dim > 600:
            scale = 600.0 / max_dim
            from PIL import Image as _Img

            small = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                _Img.Resampling.BILINEAR,
            )
            work = np.array(small)
        else:
            work = img_np

        # Binarize: dark pixels = 1
        threshold = np.mean(work)
        binary = (work < threshold).astype(np.float32)

        # Coarse sweep: -10 to +10 degrees, 0.5-degree steps
        coarse_range = np.arange(-10.0, 10.5, 0.5)
        best_angle = 0.0
        best_var = -1.0
        for angle in coarse_range:
            rotated = ndi_rotate(binary, angle, reshape=False, order=0)
            row_sums = rotated.sum(axis=1)
            var = float(np.var(row_sums))
            if var > best_var:
                best_var = var
                best_angle = angle

        # Check if the image has meaningful content (flat variance = no text)
        if best_var <= 0:
            return None

        # Fine sweep: ±1 degree around coarse estimate, 0.1-degree steps
        fine_range = np.arange(best_angle - 1.0, best_angle + 1.05, 0.1)
        for angle in fine_range:
            rotated = ndi_rotate(binary, angle, reshape=False, order=0)
            row_sums = rotated.sum(axis=1)
            var = float(np.var(row_sums))
            if var > best_var:
                best_var = var
                best_angle = angle

        return float(round(best_angle, 2))

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
        return _shared_apply(
            self,
            target=target,
            context=context,
            resolution=resolution,
            angle=angle,
            detection_resolution=detection_resolution,
            grayscale=grayscale,
            deskew_kwargs=deskew_kwargs,
        )


class HoughEngine:
    """Hough-line deskew engine (internalized from ``deskew`` package).

    Uses Canny edge detection + Hough line transform from scikit-image.

    Returns the **correction angle** in degrees — the value to pass to
    ``PIL.Image.rotate()`` to undo the detected skew (same convention as
    :class:`ProjectionProfileEngine`).
    """

    def detect(
        self,
        *,
        target: Any,
        context: Any,
        resolution: int,
        grayscale: bool,
        deskew_kwargs: Dict[str, Any],
    ) -> Optional[float]:
        from skimage.feature import canny
        from skimage.transform import hough_line, hough_line_peaks

        image = _render_target(target, resolution=resolution, grayscale=True)
        img_np: NDArray[np.uint8] = np.array(image)
        # Ensure 2-D for Canny
        if img_np.ndim == 3:
            img_np = img_np.mean(axis=2).astype(np.uint8)

        # Parameters (can be overridden via deskew_kwargs)
        sigma = deskew_kwargs.get("sigma", 3.0)
        num_peaks = deskew_kwargs.get("num_peaks", 20)
        max_skew_deg = deskew_kwargs.get("max_skew_deg", 15.0)
        min_deviation_deg = deskew_kwargs.get("min_deviation_deg", 1.0)

        # Edge detection
        edges = canny(img_np, sigma=sigma)

        # Hough transform — sweep around π/2 (the normal angle for horizontal lines).
        # skimage.transform.hough_line theta is the angle of the line's *normal*
        # from the x-axis, so a perfectly horizontal line has theta = π/2.
        num_angles = 180
        angles = np.linspace(
            np.pi / 2 - np.deg2rad(max_skew_deg),
            np.pi / 2 + np.deg2rad(max_skew_deg),
            num_angles,
            endpoint=False,
        )
        h, theta, d = hough_line(edges, theta=angles)

        # Extract peaks
        _, theta_peaks, _ = hough_line_peaks(h, theta, d, num_peaks=num_peaks)

        if len(theta_peaks) == 0:
            return None

        # Convert normal angles to baseline deviation from π/2.
        # For an image skewed +5° CCW, the line normals shift to ~(π/2 - 5°),
        # so baseline_deg ≈ -5° — which is already the correction angle
        # (rotate -5° to undo the +5° skew).
        baseline_deg = np.rad2deg(theta_peaks - np.pi / 2)
        mean_baseline = float(np.mean(baseline_deg))

        # If all detected lines are nearly horizontal, report no significant skew
        if abs(mean_baseline) < min_deviation_deg:
            return 0.0

        return float(round(mean_baseline, 2))

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
        return _shared_apply(
            self,
            target=target,
            context=context,
            resolution=resolution,
            angle=angle,
            detection_resolution=detection_resolution,
            grayscale=grayscale,
            deskew_kwargs=deskew_kwargs,
        )


def _shared_apply(
    engine,
    *,
    target: Any,
    context: Any,
    resolution: int,
    angle: Optional[float],
    detection_resolution: int,
    grayscale: bool,
    deskew_kwargs: Dict[str, Any],
) -> DeskewApplyResult:
    """Shared apply logic: detect if needed, then rotate."""
    rotation_angle = angle
    if rotation_angle is None:
        rotation_angle = engine.detect(
            target=target,
            context=context,
            resolution=detection_resolution,
            grayscale=grayscale,
            deskew_kwargs=deskew_kwargs,
        )
    image: Image.Image = _render_target(target, resolution=resolution, grayscale=False)
    if rotation_angle is None or abs(rotation_angle) <= 0.05:
        return DeskewApplyResult(image=image, angle=rotation_angle)
    if image.mode == "RGB":
        fill = (255, 255, 255)
    elif image.mode == "RGBA":
        fill = (255, 255, 255, 255)
    else:
        fill = 255
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


try:  # Register built-in engines
    register_deskew_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register deskew engines")
