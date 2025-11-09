from __future__ import annotations

from PIL import Image

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.deskew import DeskewApplyResult
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
