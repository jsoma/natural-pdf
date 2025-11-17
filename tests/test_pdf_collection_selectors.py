from natural_pdf.core.pdf_collection import PDFCollection


def _close_collection(collection: PDFCollection) -> None:
    for pdf in collection._pdfs:
        try:
            pdf.close()
        except Exception:
            pass


def test_pdf_collection_find_text():
    collection = PDFCollection(["pdfs/01-practice.pdf"])
    try:
        element = collection.find(text="Durham")
        assert element is not None
        assert "Durham" in getattr(element, "text", "")
    finally:
        _close_collection(collection)


def test_pdf_collection_find_all_across_members():
    collection = PDFCollection(["pdfs/01-practice.pdf", "pdfs/01-practice.pdf"])
    try:
        results = collection.find_all(text="Durham")
        assert len(results.elements) >= 2
    finally:
        _close_collection(collection)
