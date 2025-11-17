from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from natural_pdf.classification.classification_provider import (
    get_classification_engine,
    run_classification_batch,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasElements(Protocol):
    @property
    def elements(self) -> Sequence[Any]: ...


class ClassificationBatchMixin(_HasElements):
    def classify_all(
        self,
        labels: List[str],
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

        first_element = self.elements[0]
        engine_name = kwargs.pop("classification_engine", None)
        engine_obj = get_classification_engine(first_element, engine_name)
        inferred_using = engine_obj.infer_using(model or engine_obj.default_model("text"), using)

        items_to_classify: List[Any] = []
        original_elements: List[Any] = []
        for element in self.elements:
            if not hasattr(element, "_get_classification_content"):
                raise TypeError(f"Element {element!r} does not support classification")
            content = element._get_classification_content(model_type=inferred_using, **kwargs)
            items_to_classify.append(content)
            original_elements.append(element)

        if not items_to_classify:
            raise ValueError("No content could be gathered from elements for batch classification.")

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
            engine_name=engine_name,
            **kwargs,
        )

        if len(batch_results) != len(original_elements):
            logger.error(
                f"Batch classification result count ({len(batch_results)}) mismatch with elements processed ({len(original_elements)})."
            )
            return self

        for element, result_obj in zip(original_elements, batch_results):
            if not hasattr(element, "analyses") or element.analyses is None:
                element.analyses = {}
            element.analyses[analysis_key] = result_obj

        return self
