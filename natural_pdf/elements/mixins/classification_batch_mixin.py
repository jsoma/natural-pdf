from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from natural_pdf.classification.classification_provider import run_classification_batch

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasElements(Protocol):
    @property
    def elements(self) -> Sequence[Any]: ...


class ClassificationBatchMixin(_HasElements):
    def classify_all(
        self,
        labels: List[str],
        *,
        model: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        analysis_key: str = "classification",
        multi_label: bool = False,
        batch_size: int = 8,
        progress_bar: bool = True,
        **kwargs,
    ):
        if not getattr(self, "elements", None):
            logger.info("ElementCollection is empty, skipping classification.")
            return self

        from natural_pdf.services.classification_service import checkout_classification_engine

        first_element = self.elements[0]
        engine_name = kwargs.pop("classification_engine", None)

        # Check the engine out once and pass the instance to
        # run_classification_batch below, so exactly one engine instance is
        # created regardless of registration lifetime — and transient
        # instances are cleaned up when the call finishes.
        with checkout_classification_engine(first_element, engine_name) as engine_obj:
            inferred_using = engine_obj.infer_using(
                model or engine_obj.default_model("text"), using
            )

            items_to_classify: List[Any] = []
            original_elements: List[Any] = []
            for element in self.elements:
                if not hasattr(element, "_get_classification_content"):
                    raise TypeError(f"Element {element!r} does not support classification")
                content = element._get_classification_content(model_type=inferred_using, **kwargs)
                items_to_classify.append(content)
                original_elements.append(element)

            if not items_to_classify:
                raise ValueError(
                    "No content could be gathered from elements for batch classification."
                )

            batch_results = run_classification_batch(
                context=first_element,
                contents=items_to_classify,
                labels=labels,
                model_id=model or engine_obj.default_model(inferred_using),
                using=inferred_using,
                min_confidence=min_confidence,
                multi_label=multi_label,
                batch_size=batch_size,
                progress_bar=progress_bar,
                engine=engine_obj,
                **kwargs,
            )

        if len(batch_results) != len(original_elements):
            from natural_pdf.exceptions import ClassificationError

            raise ClassificationError(
                f"Batch classification returned {len(batch_results)} results "
                f"for {len(original_elements)} elements."
            )

        for element, result_obj in zip(original_elements, batch_results):
            if not hasattr(element, "analyses") or element.analyses is None:
                element.analyses = {}
            element.analyses[analysis_key] = result_obj

        return self
