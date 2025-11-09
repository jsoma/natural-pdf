from natural_pdf.engine_registry import register_qa_engine
from natural_pdf.qa.document_qa import DocumentQA, get_qa_engine
from natural_pdf.qa.qa_provider import register_qa_engines, run_document_qa
from natural_pdf.qa.qa_result import QAResult

__all__ = [
    "DocumentQA",
    "QAResult",
    "get_qa_engine",
    "register_qa_engine",
    "register_qa_engines",
    "run_document_qa",
]
