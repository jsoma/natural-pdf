from __future__ import annotations

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.classification.results import CategoryScore, ClassificationResult
from natural_pdf.engine_provider import EngineProvider


def test_page_classify_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubClassificationEngine:
        def infer_using(self, model_id, using):
            return using or "text"

        def default_model(self, using):
            return "stub-model"

        def classify_item(self, **kwargs):
            return ClassificationResult(
                scores=[CategoryScore("stub", 0.9)],
                model_id=kwargs.get("model_id", "stub-model"),
                using=kwargs.get("using", "text"),
            )

        def classify_batch(self, **kwargs):
            return [
                self.classify_item(model_id=kwargs.get("model_id"), using=kwargs.get("using"))
                for _ in kwargs["item_contents"]
            ]

    provider.register(
        "classification", "default", lambda **_: _StubClassificationEngine(), replace=True
    )

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    page.classify(labels=["stub"])
    result = page.analyses["classification"]
    assert result.category == "stub"
    pdf.close()
