from __future__ import annotations

import logging
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from natural_pdf.core.qa_mixin import QuestionInput
from natural_pdf.qa.qa_provider import run_document_qa
from natural_pdf.qa.qa_result import QAResult
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class QAService:
    """Service that powers Page/Region/FlowRegion question answering."""

    def __init__(self, context):
        self._context = context

    @register_delegate("qa", "ask")
    def ask(
        self,
        host: Any,
        question: QuestionInput,
        min_confidence: float = 0.1,
        model: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> Any:
        segments = self._segments(host)
        if segments is not None:
            return self._ask_segments(
                host,
                segments,
                question,
                min_confidence=min_confidence,
                model=model,
                debug=debug,
                **kwargs,
            )

        return self._ask_single(
            host,
            question,
            min_confidence=min_confidence,
            model=model,
            debug=debug,
            **kwargs,
        )

    def _ask_single(
        self,
        host: Any,
        question: QuestionInput,
        *,
        min_confidence: float,
        model: Optional[str],
        debug: bool,
        **kwargs,
    ) -> Any:
        target_region = self._target_region(host)
        try:
            raw_result = run_document_qa(
                context=host,
                region=target_region,
                question=question,
                min_confidence=min_confidence,
                model_name=model,
                debug=debug,
                **kwargs,
            )
        except ImportError as exc:
            message = (
                "Question answering requires the 'natural_pdf.qa' extras. "
                'Install with `pip install "natural-pdf[qa]"`.'
            )
            raise RuntimeError(message) from exc
        return self._normalize(host, raw_result)

    def _ask_segments(
        self,
        host: Any,
        segments: Sequence[Any],
        question: QuestionInput,
        *,
        min_confidence: float,
        model: Optional[str],
        debug: bool,
        **kwargs,
    ) -> Any:
        segment_list = list(segments)
        if not segment_list:
            blank = self._blank_result(host, question)
            return self._normalize(host, blank)

        if len(segment_list) == 1:
            return self._ask_single(
                host,
                question,
                min_confidence=min_confidence,
                model=model,
                debug=debug,
                **kwargs,
            )

        questions, return_sequence = self._coerce_questions(question)
        aggregated: List[Any] = []

        for single in questions:
            best_conf = float("-inf")
            best_result: Optional[Any] = None
            for region in segment_list:
                try:
                    candidate = self.ask(
                        region,
                        question=single,
                        min_confidence=min_confidence,
                        model=model,
                        debug=debug,
                        **kwargs,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "QA segment evaluation failed for %s: %s",
                        getattr(region, "bbox", None),
                        exc,
                    )
                    continue
                candidate_norm = self._normalize(host, candidate)
                confidence = self._confidence(host, candidate_norm)
                if confidence > best_conf:
                    best_conf = confidence
                    best_result = candidate_norm

            if best_result is None:
                blank = self._blank_result(host, single)
                best_result = self._normalize(host, blank)
            aggregated.append(best_result)

        if return_sequence:
            return aggregated
        return aggregated[0]

    def _segments(self, host: Any) -> Optional[Sequence[Any]]:
        getter = getattr(host, "_qa_segments", None)
        if callable(getter):
            try:
                segments = getter()
            except Exception:  # pragma: no cover - host bug
                return None
            if segments is None:
                return None
            if isinstance(segments, SequenceABC):
                return segments
            if isinstance(segments, IterableABC):
                return tuple(segments)
            return None
        return None

    def _target_region(self, host: Any) -> Any:
        getter = getattr(host, "_qa_target_region", None)
        if callable(getter):
            return getter()
        return host

    def _context_page_number(self, host: Any) -> int:
        getter = getattr(host, "_qa_context_page_number", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, (int, float, str)):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return -1
                return -1
            except Exception:
                pass
        page = getattr(host, "page", None)
        if page is not None:
            number = getattr(page, "number", None)
            if number is not None:
                try:
                    return int(number)
                except Exception:
                    return -1
        return -1

    def _source_elements(self, host: Any):
        getter = getattr(host, "_qa_source_elements", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
        try:
            from natural_pdf.elements.element_collection import ElementCollection
        except Exception:
            return []
        return ElementCollection([])

    def _blank_result(self, host: Any, question: QuestionInput) -> Any:
        getter = getattr(host, "_qa_blank_result", None)
        if callable(getter):
            return getter(question)

        def _build(q: str) -> QAResult:
            result = QAResult(
                question=q,
                answer="",
                confidence=0.0,
                found=False,
                page_num=self._context_page_number(host),
            )
            result.source_elements = self._source_elements(host)
            return result

        if isinstance(question, (list, tuple)):
            return [_build(str(q)) for q in question]
        return _build(str(question))

    def _normalize(self, host: Any, result: Any) -> Any:
        normalizer = getattr(host, "_qa_normalize_result", None)
        if callable(normalizer):
            return normalizer(result)
        return result

    def _confidence(self, host: Any, candidate: Any) -> float:
        getter = getattr(host, "_qa_confidence", None)
        if callable(getter):
            try:
                value = getter(candidate)
                if isinstance(value, (int, float, str)):
                    return float(value)
                return float("-inf")
            except Exception:
                return float("-inf")
        return self._default_confidence(candidate)

    def _default_confidence(self, candidate: Any) -> float:
        if isinstance(candidate, list) and candidate:
            return self._default_confidence(candidate[0])
        if isinstance(candidate, Mapping):
            value = candidate.get("confidence")
            if isinstance(value, (int, float, str)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("-inf")
            return float("-inf")
        value = getattr(candidate, "confidence", None)
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")
        return float("-inf")

    @staticmethod
    def _coerce_questions(question: QuestionInput) -> Tuple[List[str], bool]:
        if isinstance(question, (list, tuple)):
            return [str(q) for q in question], True
        return [str(question)], False
