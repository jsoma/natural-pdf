from __future__ import annotations

import numpy as np
from PIL import Image

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.deskew import DeskewApplyResult, HoughEngine, ProjectionProfileEngine
from natural_pdf.engine_provider import EngineProvider


def test_page_detect_skew_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubDeskew:
        def detect(self, **kwargs):
            return 1.23

        def apply(self, **kwargs):  # pragma: no cover - not used here
            return DeskewApplyResult(image=None, angle=kwargs.get("angle", 0))

    engine = _StubDeskew()
    provider.register("deskew.detect", "standard", lambda **_: engine, replace=True)
    provider.register("deskew", "standard", lambda **_: engine, replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    angle = page.detect_skew_angle(resolution=10, force_recalculate=True)
    assert angle == 1.23
    pdf.close()


def test_page_deskew_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubDeskew:
        def detect(self, **kwargs):
            return kwargs.get("deskew_kwargs", {}).get("fallback", 0.0)

        def apply(self, **kwargs):
            img = Image.new("RGB", (10, 10), color="white")
            ang = kwargs.get("angle")
            if ang is None:
                ang = self.detect(**kwargs)
            return DeskewApplyResult(image=img, angle=ang)

    engine = _StubDeskew()
    provider.register("deskew.apply", "standard", lambda **_: engine, replace=True)
    provider.register("deskew.detect", "standard", lambda **_: engine, replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    image = page.deskew(angle=5.0)
    assert image.size == (10, 10)

    image2 = page.deskew(angle=None, deskew_kwargs={"fallback": 2.0})
    assert image2.size == (10, 10)
    pdf.close()


def test_page_detect_skew_engine_param(monkeypatch):
    """Passing engine= routes to the correct named engine."""
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubProjection:
        def detect(self, **kwargs):
            return 1.0

    class _StubHough:
        def detect(self, **kwargs):
            return 2.0

    proj = _StubProjection()
    hough = _StubHough()
    provider.register("deskew.detect", "projection", lambda **_: proj, replace=True)
    provider.register("deskew.detect", "hough", lambda **_: hough, replace=True)
    # Also register "standard" -> projection (matches real registration)
    provider.register("deskew.detect", "standard", lambda **_: proj, replace=True)
    provider.register("deskew", "standard", lambda **_: proj, replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Default (no engine) -> standard -> projection stub -> 1.0
    assert page.detect_skew_angle(resolution=10, force_recalculate=True) == 1.0

    # Explicit projection
    assert page.detect_skew_angle(resolution=10, force_recalculate=True, engine="projection") == 1.0

    # Explicit hough
    assert page.detect_skew_angle(resolution=10, force_recalculate=True, engine="hough") == 2.0

    pdf.close()


def test_page_deskew_engine_param(monkeypatch):
    """Passing engine= to page.deskew routes correctly."""
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubEngine:
        def __init__(self, marker):
            self.marker = marker

        def apply(self, **kwargs):
            img = Image.new("RGB", (self.marker, self.marker), color="white")
            return DeskewApplyResult(image=img, angle=0.5)

    proj = _StubEngine(10)
    hough = _StubEngine(20)

    def _proj_factory(**_):
        return proj

    def _hough_factory(**_):
        return hough

    for cap in ("deskew", "deskew.apply", "deskew.detect"):
        provider.register(cap, "standard", _proj_factory, replace=True)
        provider.register(cap, "projection", _proj_factory, replace=True)
        provider.register(cap, "hough", _hough_factory, replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    img_default = page.deskew(angle=1.0)
    assert img_default.size == (10, 10)

    img_hough = page.deskew(angle=1.0, engine="hough")
    assert img_hough.size == (20, 20)

    pdf.close()


# --- Integration tests using real engines on synthetic images ---


def _make_skewed_image(angle_deg: float, width: int = 400, height: int = 300) -> Image.Image:
    """Create a synthetic image with horizontal text lines, rotated by angle_deg."""
    img = np.ones((height, width), dtype=np.uint8) * 255
    # Draw horizontal black lines every 20px
    for y in range(20, height - 20, 20):
        img[y : y + 2, 40 : width - 40] = 0
    pil_img = Image.fromarray(img, mode="L")
    # Rotate by the given angle (positive = counter-clockwise)
    rotated = pil_img.rotate(
        angle_deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255
    )
    return rotated


class _FakeTarget:
    """Minimal target that satisfies _render_target()."""

    def __init__(self, image: Image.Image):
        self._image = image

    def render(self, resolution: int = 72) -> Image.Image:
        return self._image.copy()


def test_projection_engine_detects_known_skew():
    """ProjectionProfileEngine returns a correction angle (opposite sign to applied skew)."""
    known_skew = 3.0  # Image rotated +3° CCW
    img = _make_skewed_image(known_skew)
    target = _FakeTarget(img)
    engine = ProjectionProfileEngine()
    detected = engine.detect(
        target=target, context=None, resolution=72, grayscale=True, deskew_kwargs={}
    )
    assert detected is not None
    # Correction angle should be approximately -known_skew
    assert abs(detected + known_skew) < 1.5, f"Expected ~-{known_skew}, got {detected}"


def test_projection_engine_no_skew():
    """ProjectionProfileEngine returns ~0 for an upright image."""
    img = _make_skewed_image(0.0)
    target = _FakeTarget(img)
    engine = ProjectionProfileEngine()
    detected = engine.detect(
        target=target, context=None, resolution=72, grayscale=True, deskew_kwargs={}
    )
    assert detected is not None
    assert abs(detected) < 1.0, f"Expected ~0, got {detected}"


def test_hough_engine_detects_skew():
    """HoughEngine returns a correction angle (opposite sign to applied skew)."""
    known_skew = 5.0  # Image rotated +5° CCW
    img = _make_skewed_image(known_skew)
    target = _FakeTarget(img)
    engine = HoughEngine()
    detected = engine.detect(
        target=target, context=None, resolution=72, grayscale=True, deskew_kwargs={}
    )
    assert detected is not None, "HoughEngine should detect lines in the synthetic image"
    # Correction angle should be approximately -known_skew
    assert abs(detected + known_skew) < 3.0, f"Expected ~-{known_skew}, got {detected}"


def test_projection_engine_apply():
    """ProjectionProfileEngine.apply() returns a DeskewApplyResult with an image."""
    img = _make_skewed_image(2.0)
    target = _FakeTarget(img)
    engine = ProjectionProfileEngine()
    result = engine.apply(
        target=target,
        context=None,
        resolution=72,
        angle=2.0,
        detection_resolution=72,
        grayscale=True,
        deskew_kwargs={},
    )
    assert isinstance(result, DeskewApplyResult)
    assert result.image is not None
    assert result.angle == 2.0


def test_hough_engine_apply():
    """HoughEngine.apply() returns a DeskewApplyResult with an image."""
    img = _make_skewed_image(2.0)
    target = _FakeTarget(img)
    engine = HoughEngine()
    result = engine.apply(
        target=target,
        context=None,
        resolution=72,
        angle=2.0,
        detection_resolution=72,
        grayscale=True,
        deskew_kwargs={},
    )
    assert isinstance(result, DeskewApplyResult)
    assert result.image is not None
    assert result.angle == 2.0


def test_default_engine_is_projection(monkeypatch):
    """When no engine is specified, the standard registration uses ProjectionProfileEngine."""
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    # Re-register using the real function
    from natural_pdf.deskew.deskew_provider import register_deskew_engines

    register_deskew_engines(provider)

    engine = provider.get("deskew.detect", context=None, name="standard")
    assert isinstance(engine, ProjectionProfileEngine)
