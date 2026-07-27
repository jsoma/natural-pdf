"""URL downloads must send an identifying User-Agent.

Some hosts (e.g. Cloudflare R2) return 403 to the default Python-urllib
User-Agent while serving the same URL to browsers and curl.
"""

from __future__ import annotations

import io
import urllib.request

import pytest
from PIL import Image

import natural_pdf as npdf

TEST_PDF = "pdfs/01-practice.pdf"


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture_urlopen(monkeypatch, payload: bytes):
    captured = {}

    def fake_urlopen(request, *args, **kwargs):
        captured["request"] = request
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _assert_natural_pdf_user_agent(request, expected_url: str):
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == expected_url
    user_agent = request.get_header("User-agent")
    assert user_agent is not None, "download request sent no User-Agent"
    assert user_agent.startswith("natural-pdf/")
    assert "github.com/jsoma/natural-pdf" in user_agent


def test_pdf_url_download_sends_user_agent(monkeypatch):
    with open(TEST_PDF, "rb") as fh:
        pdf_bytes = fh.read()

    captured = _capture_urlopen(monkeypatch, pdf_bytes)

    pdf = npdf.PDF("https://example.com/sample.pdf")
    try:
        _assert_natural_pdf_user_agent(captured["request"], "https://example.com/sample.pdf")
        assert len(pdf.pages) > 0
    finally:
        pdf.close()


def test_from_images_url_download_sends_user_agent(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), "white").save(buffer, format="PNG")

    captured = _capture_urlopen(monkeypatch, buffer.getvalue())

    pdf = npdf.PDF.from_images("https://example.com/scan.png", apply_ocr=False)
    try:
        _assert_natural_pdf_user_agent(captured["request"], "https://example.com/scan.png")
        assert len(pdf.pages) == 1
    finally:
        pdf.close()
