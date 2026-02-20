# Internal-only exports used by extraction_service.py for the doc_qa engine.
# These are NOT part of the public API.
from natural_pdf.qa.document_qa import DocumentQA, get_qa_engine

__all__ = [
    "DocumentQA",
    "get_qa_engine",
]
