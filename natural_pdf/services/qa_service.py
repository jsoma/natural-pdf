from __future__ import annotations

import logging
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import Field, create_model

from natural_pdf.core.qa_mixin import QuestionInput
from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)

# Cached one-field schema used by _ask_single and _blank_structured_result.
_QA_SCHEMA = create_model("_QA", answer=(Optional[str], Field(None)))


class QAService:
    """Service that powers Page/Region/FlowRegion question answering.

    Delegates to ``host.extract()`` under the hood, returning
    :class:`StructuredDataResult` in all cases.
    """

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
        *,
        client: Any = None,
        using: str = "text",
        engine: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Ask a question about *host* and return a :class:`StructuredDataResult`.

        New keyword arguments ``client``, ``using``, and ``engine`` are
        forwarded to ``.extract()``.  When none of them are supplied the
        default doc-QA (LayoutLM) engine is used.
        """
        # Handle batch questions
        questions, return_sequence = self._coerce_questions(question)
        if return_sequence:
            results = [
                self._dispatch_single(
                    host,
                    q,
                    min_confidence=min_confidence,
                    model=model,
                    debug=debug,
                    client=client,
                    using=using,
                    engine=engine,
                    **kwargs,
                )
                for q in questions
            ]
            return results

        return self._dispatch_single(
            host,
            questions[0],
            min_confidence=min_confidence,
            model=model,
            debug=debug,
            client=client,
            using=using,
            engine=engine,
            **kwargs,
        )

    def _dispatch_single(
        self,
        host: Any,
        question: str,
        *,
        min_confidence: float,
        model: Optional[str],
        debug: bool,
        client: Any = None,
        using: str = "text",
        engine: Optional[str] = None,
        **kwargs,
    ) -> StructuredDataResult:
        """Route a single question through segments (if any) or directly."""
        segments = self._segments(host)
        if segments is not None:
            if len(segments) > 1:
                return self._ask_segments(
                    host,
                    segments,
                    question,
                    min_confidence=min_confidence,
                    model=model,
                    debug=debug,
                    client=client,
                    using=using,
                    engine=engine,
                    **kwargs,
                )
            if len(segments) == 1:
                # Unwrap single-segment hosts (e.g. PageCollection with 1 page)
                return self._ask_single(
                    segments[0],
                    question,
                    min_confidence=min_confidence,
                    model=model,
                    debug=debug,
                    client=client,
                    using=using,
                    engine=engine,
                    **kwargs,
                )
            # Empty segments
            return self._blank_structured_result(question)

        return self._ask_single(
            host,
            question,
            min_confidence=min_confidence,
            model=model,
            debug=debug,
            client=client,
            using=using,
            engine=engine,
            **kwargs,
        )

    def _ask_single(
        self,
        host: Any,
        question: str,
        *,
        min_confidence: float,
        model: Optional[str],
        debug: bool,
        client: Any = None,
        using: str = "text",
        engine: Optional[str] = None,
        **kwargs,
    ) -> StructuredDataResult:
        """Create a one-field schema and call ``host.extract()``."""
        extract_kw: dict[str, Any] = dict(
            schema=_QA_SCHEMA,
            model=model,
            analysis_key="_qa",
            overwrite=True,
            citations=kwargs.pop("citations", False),
        )
        if debug:
            extract_kw["debug"] = True

        if client is not None:
            extract_kw.update(
                client=client,
                using=using,
                prompt=(
                    "Answer this question about the document, or null if not found.\n\n"
                    f"Question: {question}"
                ),
            )
        elif engine == "vlm":
            extract_kw.update(
                engine="vlm",
                using="vision",
                prompt=(
                    "Answer this question about the document image, or null if not found.\n\n"
                    f"Question: {question}"
                ),
            )
        else:
            extract_kw.update(
                engine="doc_qa",
                question_map={"answer": str(question)},
                min_confidence=min_confidence,
            )

        extract_kw.update(kwargs)

        # Use the extract service via the host's .extract() method.
        # If host doesn't have .extract(), try its _qa_target_region().
        extract_host = host
        if not hasattr(extract_host, "extract"):
            target_getter = getattr(host, "_qa_target_region", None)
            if callable(target_getter):
                extract_host = target_getter()
        extract_fn = getattr(extract_host, "extract", None)
        if extract_fn is None:
            raise RuntimeError(
                f"{type(host).__name__} does not support .extract(); " "cannot delegate .ask()."
            )

        try:
            return extract_fn(**extract_kw)
        except ImportError as exc:
            message = (
                "Question answering requires the 'natural_pdf.qa' extras. "
                'Install with `pip install "natural-pdf[qa]"`.'
            )
            raise RuntimeError(message) from exc

    def _ask_segments(
        self,
        host: Any,
        segments: Sequence[Any],
        question: str,
        *,
        min_confidence: float,
        model: Optional[str],
        debug: bool,
        client: Any = None,
        using: str = "text",
        engine: Optional[str] = None,
        **kwargs,
    ) -> StructuredDataResult:
        """Try each segment, pick the result with the best confidence."""
        segment_list = list(segments)
        if not segment_list:
            return self._blank_structured_result(question)

        if len(segment_list) == 1:
            return self._ask_single(
                host,
                question,
                min_confidence=min_confidence,
                model=model,
                debug=debug,
                client=client,
                using=using,
                engine=engine,
                **kwargs,
            )

        best_conf = float("-inf")
        best_result: Optional[StructuredDataResult] = None

        for region in segment_list:
            try:
                candidate = self._ask_single(
                    region,
                    question,
                    min_confidence=min_confidence,
                    model=model,
                    debug=debug,
                    client=client,
                    using=using,
                    engine=engine,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "QA segment evaluation failed for %s: %s",
                    getattr(region, "bbox", None),
                    exc,
                )
                continue

            confidence = self._extract_confidence(candidate)
            if confidence > best_conf:
                best_conf = confidence
                best_result = candidate

        if best_result is None:
            return self._blank_structured_result(question)
        return best_result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
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

    @staticmethod
    def _extract_confidence(result: StructuredDataResult) -> float:
        """Extract confidence from a StructuredDataResult.

        Checks two sources in order:
        1. Per-field confidence from ``result._fields["answer"].confidence``
        2. ``data.answer_confidence`` attribute (doc_qa path)

        Returns ``0.0`` when an answer is present but no confidence data
        is available, or ``-inf`` when there is no answer at all.
        """
        if not result.success or result.data is None:
            return float("-inf")

        # Source 1: per-field confidence from _fields
        fields = getattr(result, "_fields", {})
        if "answer" in fields:
            conf = fields["answer"].confidence
            if conf is not None:
                try:
                    return float(conf)
                except (TypeError, ValueError):
                    pass

        # Source 2: data.answer_confidence attribute (doc_qa path)
        data = result.data
        conf = getattr(data, "answer_confidence", None)
        if conf is not None:
            try:
                return float(conf)
            except (TypeError, ValueError):
                pass

        # Fallback: answer present but unknown confidence → 0.0
        answer = getattr(data, "answer", None)
        if answer is not None and str(answer).strip():
            return 0.0

        return float("-inf")

    @staticmethod
    def _blank_structured_result(_question: str = "") -> StructuredDataResult:
        """Return a failed StructuredDataResult for unanswered questions."""
        instance = _QA_SCHEMA(answer=None)
        return StructuredDataResult(
            data=instance,
            success=False,
            error_message="No content available to answer the question.",
            raw_output=None,
            model_used=None,
        )

    @staticmethod
    def _coerce_questions(question: QuestionInput) -> Tuple[List[str], bool]:
        if isinstance(question, (list, tuple)):
            return [str(q) for q in question], True
        return [str(question)], False
