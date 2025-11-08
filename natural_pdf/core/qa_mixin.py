from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from natural_pdf.qa.qa_result import QAResult

logger = logging.getLogger(__name__)

QuestionInput = Union[str, Sequence[str], Tuple[str, ...]]


class DocumentQAMixin:
    """Shared QA helpers for Pages, Regions, FlowRegions."""

    def ask(
        self,
        question: QuestionInput,
        min_confidence: float = 0.1,
        model: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> Any:
        try:
            from natural_pdf.qa.document_qa import get_qa_engine
        except ImportError:
            logger.error(
                "Question answering requires the 'natural_pdf.qa' extras. Install with `pip install \"natural-pdf[ai]\"`."
            )
            return self._qa_blank_result(question)

        try:
            qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
        except Exception as exc:
            logger.error("Failed to initialize QA engine: %s", exc, exc_info=True)
            return self._qa_blank_result(question)

        try:
            target_region = self._qa_target_region()
            raw_result = qa_engine.ask_pdf_region(
                target_region,
                question,
                min_confidence=min_confidence,
                debug=debug,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Error running document QA: %s", exc, exc_info=True)
            return self._qa_blank_result(question)

        return self._qa_normalize_result(raw_result)

    def _qa_context_page_number(self) -> int:
        raise NotImplementedError

    def _qa_source_elements(self) -> "ElementCollection":
        from natural_pdf.elements.element_collection import ElementCollection

        return ElementCollection([])

    def _qa_blank_result(self, question: QuestionInput) -> Union[QAResult, List[QAResult]]:
        def _build(q: str) -> QAResult:
            result = QAResult(
                question=q,
                answer="",
                confidence=0.0,
                found=False,
                page_num=self._qa_context_page_number(),
            )
            result.source_elements = self._qa_source_elements()
            return result

        if isinstance(question, (list, tuple)):
            return [_build(str(q)) for q in question]
        return _build(str(question))

    def _qa_normalize_result(self, result: Any) -> Any:
        return result

    def _qa_target_region(self) -> Any:
        raise NotImplementedError


if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection
