"""Regression coverage for breaking contract cleanup and public metadata."""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

import natural_pdf
from natural_pdf import PDF
from natural_pdf.core.word_engine import WordEngineOptions


def test_dead_document_text_order_controls_are_removed() -> None:
    signature = inspect.signature(PDF)
    assert "reading_order" not in signature.parameters
    assert signature.parameters["font_attrs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "use_text_flow" not in {field.name for field in fields(WordEngineOptions)}

    with pytest.raises(TypeError, match="reading_order"):
        PDF("unused.pdf", reading_order=True)


def test_set_option_is_part_of_the_root_public_api() -> None:
    assert natural_pdf.set_option is not None
    assert "set_option" in natural_pdf.__all__
