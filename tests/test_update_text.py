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


def test_update_text_bumps_text_state_only_when_content_changes():
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]
    initial_version = page._text_state_version

    def identity(element):
        return element.text

    def append_marker(element):
        return f"{element.text}!"

    try:
        page.update_text(identity)
        assert page._text_state_version == initial_version

        page.update_text(append_marker)
        assert page._text_state_version > initial_version
    finally:
        pdf.close()
