import pytest

from natural_pdf import PDF


def test_update_text_hello(tmp_path):
    """Ensure that update_text replaces every text element with 'hello'."""

    pdf_path = "pdfs/01-practice.pdf"

    pdf = PDF(pdf_path)

    def to_hello(_):
        return "hello"

    # run update_text across entire document
    pdf.update_text(to_hello)

    # Verify
    for page in pdf.pages:
        for el in page.find_all("text").elements:
            assert el.text == "hello"
