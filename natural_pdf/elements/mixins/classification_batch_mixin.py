from __future__ import annotations

import logging
from typing import Any, List, Optional

from natural_pdf.classification.classification_provider import (
    get_classification_engine,
    run_classification_batch,
)
from natural_pdf.classification.mixin import ClassificationMixin

logger = logging.getLogger(__name__)


class ClassificationBatchMixin:
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
            if not isinstance(element, ClassificationMixin):
                logger.warning(f"Skipping element (not ClassificationMixin): {element!r}")
                continue
            try:
                content = element._get_classification_content(model_type=inferred_using, **kwargs)
                items_to_classify.append(content)
                original_elements.append(element)
            except Exception as exc:
                logger.warning(f"Skipping element {element!r}: {exc}")

        if not items_to_classify:
            logger.warning("No content could be gathered from elements for batch classification.")
            return self

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
            try:
                if not hasattr(element, "analyses") or element.analyses is None:
                    element.analyses = {}
                element.analyses[analysis_key] = result_obj
            except Exception as exc:
                logger.warning(f"Failed to store classification result for {element!r}: {exc}")

        return self
