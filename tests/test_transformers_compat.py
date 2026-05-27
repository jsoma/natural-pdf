"""Tests for transformers compatibility helpers."""

from __future__ import annotations

import sys
import types

import pytest

from natural_pdf.utils.transformers_compat import get_pp_doclayout_v3_classes


class NewProcessor:
    pass


class FastProcessor:
    pass


class Model:
    pass


def _mock_transformers(monkeypatch, **attrs):
    module = types.ModuleType("transformers")
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_pp_doclayout_prefers_unsuffixed_processor(monkeypatch):
    _mock_transformers(
        monkeypatch,
        PPDocLayoutV3ForObjectDetection=Model,
        PPDocLayoutV3ImageProcessor=NewProcessor,
        PPDocLayoutV3ImageProcessorFast=FastProcessor,
    )

    processor_cls, model_cls = get_pp_doclayout_v3_classes()

    assert processor_cls is NewProcessor
    assert model_cls is Model


def test_pp_doclayout_falls_back_to_fast_processor(monkeypatch):
    _mock_transformers(
        monkeypatch,
        PPDocLayoutV3ForObjectDetection=Model,
        PPDocLayoutV3ImageProcessorFast=FastProcessor,
    )

    processor_cls, model_cls = get_pp_doclayout_v3_classes()

    assert processor_cls is FastProcessor
    assert model_cls is Model


def test_pp_doclayout_requires_processor(monkeypatch):
    _mock_transformers(monkeypatch, PPDocLayoutV3ForObjectDetection=Model)

    with pytest.raises(ImportError, match="PPDocLayoutV3ImageProcessor"):
        get_pp_doclayout_v3_classes()
