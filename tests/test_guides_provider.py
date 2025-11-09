import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.analyzers.guides.base import Guides
from natural_pdf.engine_provider import EngineProvider


def test_guides_from_content_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubGuidesEngine:
        def detect(self, axis, method, context, options):
            assert method == "content"
            return type("Result", (), {"coordinates": [1.0, 2.0]})()

    provider.register("guides.detect", "builtin", lambda **_: _StubGuidesEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    guides = Guides(context=pdf.pages[0])
    guides.vertical.from_content("text")
    assert list(guides.vertical.data) == [1.0, 2.0]
    pdf.close()


def test_guides_from_headers_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubGuidesEngine:
        def detect(self, axis, method, context, options):
            assert axis == "vertical"
            assert method == "headers"
            assert options["headers"] == ["Name"]
            return type("Result", (), {"coordinates": [0.0, 100.0, 200.0]})()

    provider.register("guides.detect", "builtin", lambda **_: _StubGuidesEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    guides = Guides(context=pdf.pages[0])
    guides.vertical.from_headers(["Name"])
    assert list(guides.vertical.data) == [0.0, 100.0, 200.0]
    pdf.close()


def test_guides_from_stripes_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubGuidesEngine:
        def detect(self, axis, method, context, options):
            assert axis == "horizontal"
            assert method == "stripes"
            assert options["color"] == "#00ffff"
            return type("Result", (), {"coordinates": [50.0, 75.0, 100.0]})()

    provider.register("guides.detect", "builtin", lambda **_: _StubGuidesEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    guides = Guides(context=pdf.pages[0])
    guides.horizontal.from_stripes(color="#00ffff")
    assert list(guides.horizontal.data) == [50.0, 75.0, 100.0]
    pdf.close()
