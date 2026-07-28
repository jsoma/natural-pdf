import urllib.request

import pytest

import natural_pdf as npdf
from natural_pdf.exporters import original_pdf


def test_url_password_error_is_not_misreported_as_download_failure(monkeypatch, tmp_path):
    class PasswordError(Exception):
        pass

    class FakePdf:
        @staticmethod
        def open(_source):
            raise PasswordError("password required")

    class FakePikePdf:
        Pdf = FakePdf

    FakePikePdf.PasswordError = PasswordError

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"encrypted pdf"

    captured = {}

    def fake_urlopen(request, *_args, **_kwargs):
        captured["request"] = request
        return Response()

    monkeypatch.setattr(original_pdf, "require", lambda _name: FakePikePdf)
    monkeypatch.setattr(original_pdf.urllib.request, "urlopen", fake_urlopen)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        pdf.path = "https://example.invalid/encrypted.pdf"
        pdf._original_bytes = None
        with pytest.raises(PasswordError, match="password required"):
            original_pdf.create_original_pdf(pdf.pages[0], tmp_path / "result.pdf")
        assert isinstance(captured["request"], urllib.request.Request)
        assert captured["request"].get_header("User-agent").startswith("natural-pdf/")
    finally:
        pdf.close()
