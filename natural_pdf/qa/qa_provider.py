"""Provider plumbing for document question answering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_qa_engine
from natural_pdf.qa.qa_result import QAResult

if TYPE_CHECKING:
    from natural_pdf.qa.document_qa import DocumentQA

logger = logging.getLogger(__name__)


class DocumentQAEngine:
    def ask_region(
        self,
        *,
        region: Any,
        question: Union[str, list[str], tuple[str, ...]],
        min_confidence: float,
        debug: bool,
        **kwargs,
    ) -> Union[QAResult, list[QAResult]]:
        raise NotImplementedError


class _DefaultDocumentQAEngine(DocumentQAEngine):
    def __init__(self) -> None:
        self._engine: Optional["DocumentQA"] = None
        self._model_name: Optional[str] = None

    def _ensure_engine(self, model_name: Optional[str]) -> None:
        target_model = model_name or "impira/layoutlm-document-qa"
        if self._engine is not None and self._model_name == target_model:
            return
        from natural_pdf.qa.document_qa import DocumentQA

        self._engine = DocumentQA(model_name=target_model)
        self._model_name = target_model

    def ask_region(self, *, model_name: Optional[str] = None, **kwargs):
        self._ensure_engine(model_name)
        if self._engine is None:
            raise RuntimeError("DocumentQA engine is not initialized.")
        return self._engine.ask_pdf_region(**kwargs)


def register_qa_engines(provider=None) -> None:
    def factory(**_opts):
        return _DefaultDocumentQAEngine()

    register_builtin(provider, "qa.document", "layoutlm", factory)


def run_document_qa(
    *,
    context: Any,
    region: Any,
    question: Union[str, list[str], tuple[str, ...]],
    min_confidence: float,
    model_name: Optional[str],
    debug: bool,
    engine_name: Optional[str] = None,
    **kwargs,
) -> Union[QAResult, list[QAResult]]:
    provider = get_provider()
    name = (engine_name or "layoutlm").strip().lower()
    engine = provider.get("qa.document", context=context, name=name)
    return engine.ask_region(
        region=region,
        question=question,
        min_confidence=min_confidence,
        debug=debug,
        model_name=model_name,
        **kwargs,
    )


try:
    register_qa_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register QA engines")


__all__ = ["register_qa_engines", "run_document_qa"]
