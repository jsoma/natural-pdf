"""Deskew provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)

# Correction angles at or below this magnitude (degrees) are treated as "not
# skewed" and skip the rotation entirely — rotating by a few hundredths of a
# degree only degrades the raster without visibly straightening anything.
NO_SKEW_EPSILON_DEG = 0.05


@runtime_checkable
class DeskewEngine(Protocol):
    """Structural interface for deskew engines.

    ``detect`` returns the correction angle in degrees, ``0.0`` when content
    was found but is not significantly skewed, or ``None`` when detection was
    not possible (e.g. a blank page).
    """

    def detect(
        self,
        *,
        target: Any,
        context: Any,
        resolution: int,
        grayscale: bool,
        deskew_kwargs: Dict[str, Any],
    ) -> Optional[float]: ...

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
    ) -> "DeskewApplyResult": ...


@dataclass
class DeskewApplyResult:
    image: Image.Image
    angle: Optional[float]


def _validate_deskew_kwargs(engine_label: str, kwargs: Dict[str, Any], allowed: set) -> None:
    """Reject unknown tuning keys instead of silently discarding typos."""
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(
            f"Unknown deskew_kwargs for {engine_label}: {sorted(unknown)}. "
            f"Supported keys: {sorted(allowed)}"
        )


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
    with ExitStack() as stack:
        try:
            engine = stack.enter_context(
                provider.checkout("deskew.detect", context=context, name=name)
            )
        except LookupError:
            engine = stack.enter_context(provider.checkout("deskew", context=context, name=name))
        if not isinstance(engine, DeskewEngine):
            raise TypeError(f"Deskew engine '{name}' does not implement the DeskewEngine interface")
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
    with ExitStack() as stack:
        try:
            engine = stack.enter_context(
                provider.checkout("deskew.apply", context=context, name=name)
            )
        except LookupError:
            engine = stack.enter_context(provider.checkout("deskew", context=context, name=name))
        if not isinstance(engine, DeskewEngine):
            raise TypeError(f"Deskew engine '{name}' does not implement the DeskewEngine interface")
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

    #: Tuning keys accepted via deskew_kwargs, with defaults.
    SUPPORTED_KWARGS = {
        "downsample_max_dim",  # px cap for the working image (default 600)
        "coarse_range_deg",  # coarse sweep half-range (default 10.0)
        "coarse_step_deg",  # coarse sweep step (default 0.5)
        "fine_range_deg",  # fine sweep half-range around coarse best (default 1.0)
        "fine_step_deg",  # fine sweep step (default 0.1)
    }

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

        _validate_deskew_kwargs("ProjectionProfileEngine", deskew_kwargs, self.SUPPORTED_KWARGS)
        downsample_max_dim = deskew_kwargs.get("downsample_max_dim", 600)
        coarse_range_deg = deskew_kwargs.get("coarse_range_deg", 10.0)
        coarse_step_deg = deskew_kwargs.get("coarse_step_deg", 0.5)
        fine_range_deg = deskew_kwargs.get("fine_range_deg", 1.0)
        fine_step_deg = deskew_kwargs.get("fine_step_deg", 0.1)

        image = _render_target(target, resolution=resolution, grayscale=grayscale)
        img_np: NDArray[np.uint8] = np.array(image)
        # Ensure 2-D even if render returned an RGB image (grayscale=False)
        if img_np.ndim == 3:
            img_np = img_np.mean(axis=2).astype(np.uint8)

        # Downsample for speed during coarse sweep
        max_dim = max(img_np.shape[:2])
        if max_dim > downsample_max_dim:
            scale = float(downsample_max_dim) / max_dim
            from PIL import Image as _Img

            small = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                _Img.Resampling.BILINEAR,
            )
            work = np.array(small)
            if work.ndim == 3:
                work = work.mean(axis=2).astype(np.uint8)
        else:
            work = img_np

        # Binarize: dark pixels = 1
        threshold = np.mean(work)
        binary = (work < threshold).astype(np.float32)

        # Coarse sweep
        coarse_range = np.arange(
            -coarse_range_deg, coarse_range_deg + coarse_step_deg, coarse_step_deg
        )
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

        # Fine sweep around the coarse estimate
        fine_range = np.arange(
            best_angle - fine_range_deg,
            best_angle + fine_range_deg + fine_step_deg / 2,
            fine_step_deg,
        )
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

    #: Tuning keys accepted via deskew_kwargs, with defaults.
    SUPPORTED_KWARGS = {
        "sigma",  # Canny gaussian sigma (default 3.0)
        "num_peaks",  # max Hough peaks considered (default 20)
        "max_skew_deg",  # half-range of the angle sweep (default 15.0)
        "min_deviation_deg",  # below this mean deviation, report 0.0 (default 1.0)
    }

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

        _validate_deskew_kwargs("HoughEngine", deskew_kwargs, self.SUPPORTED_KWARGS)

        image = _render_target(target, resolution=resolution, grayscale=grayscale)
        img_np: NDArray[np.uint8] = np.array(image)
        # Ensure 2-D for Canny (grayscale=False renders RGB)
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
    if rotation_angle is None or abs(rotation_angle) <= NO_SKEW_EPSILON_DEG:
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


# Register built-in engines at import time. A failure here must surface
# immediately — swallowing it turns every later deskew call into an opaque
# LookupError.
register_deskew_engines()
