import json
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from natural_pdf.elements.element_collection import ElementCollection

from .qa_result import QAResult

logger = logging.getLogger("natural_pdf.qa.document_qa")

# Global QA engine instance
_QA_ENGINE_INSTANCE = None


def get_qa_engine(model_name: str = "impira/layoutlm-document-qa", **kwargs):
    """
    Get or create a global QA engine instance.

    Args:
        model_name: Name of the model to use (default: "impira/layoutlm-document-qa")
        **kwargs: Additional parameters to pass to the DocumentQA constructor

    Returns:
        DocumentQA instance
    """
    global _QA_ENGINE_INSTANCE

    if _QA_ENGINE_INSTANCE is None:
        try:
            _QA_ENGINE_INSTANCE = DocumentQA(model_name=model_name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize QA engine: {e}")
            raise

    return _QA_ENGINE_INSTANCE


class DocumentQA:
    """
    Document Question Answering using LayoutLM.

    This class provides the ability to ask natural language questions about document content,
    leveraging the spatial layout information from PDF pages.
    """

    def __init__(self, model_name: str = "impira/layoutlm-document-qa", device: str = None):
        """
        Initialize the Document QA engine.

        Args:
            model_name: HuggingFace model name to use (default: "impira/layoutlm-document-qa")
            device: Device to run the model on ('cuda' or 'cpu'). If None, will use cuda if available.
        """
        try:
            import torch
            from transformers import pipeline

            logger.info(f"Initializing DocumentQA with model {model_name} on {device}")

            # Try MPS, fallback to CPU if OOM
            if device is None and torch.backends.mps.is_available():
                try:
                    self.pipe = pipeline(
                        "document-question-answering", model=model_name, device="mps"
                    )
                    self.device = "mps"
                except RuntimeError as e:
                    logger.warning(f"MPS OOM: {e}, falling back to CPU")
                    self.pipe = pipeline(
                        "document-question-answering", model=model_name, device="cpu"
                    )
                    self.device = "cpu"
            else:
                self.pipe = pipeline("document-question-answering", model=model_name, device=device)
                self.device = device

            self.model_name = model_name
            self.device = device
            self._is_initialized = True

        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            self._is_initialized = False
            raise ImportError(
                "DocumentQA requires transformers and torch to be installed. "
                "Install with pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DocumentQA: {e}")
            self._is_initialized = False
            raise

    def is_available(self) -> bool:
        """Check if the QA engine is properly initialized."""
        return self._is_initialized

    def _get_word_boxes_from_elements(self, elements, offset_x=0, offset_y=0) -> List[List]:
        """
        Extract word boxes from text elements.

        Args:
            elements: List of TextElement objects
            offset_x: X-coordinate offset to subtract (for region cropping)
            offset_y: Y-coordinate offset to subtract (for region cropping)

        Returns:
            List of [text, [x0, top, x1, bottom]] entries
        """
        word_boxes = []

        for element in elements:
            if hasattr(element, "text") and element.text.strip():
                # Apply offset for cropped regions
                x0 = int(element.x0) - offset_x
                top = int(element.top) - offset_y
                x1 = int(element.x1) - offset_x
                bottom = int(element.bottom) - offset_y

                # Ensure coordinates are valid (non-negative)
                x0 = max(0, x0)
                top = max(0, top)
                x1 = max(0, x1)
                bottom = max(0, bottom)

                word_boxes.append([element.text, [x0, top, x1, bottom]])

        return word_boxes

    def ask(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: Union[str, List[str], Tuple[str, ...]],
        word_boxes: List = None,
        min_confidence: float = 0.1,
        debug: bool = False,
        debug_output_dir: str = "output",
    ) -> Union[QAResult, List[QAResult]]:
        """
        Ask one or more natural-language questions about the supplied document image.

        This method now accepts a single *question* (``str``) **or** an
        iterable of questions (``list``/``tuple`` of ``str``).  When multiple
        questions are provided they are executed in a single batch through the
        underlying transformers pipeline which is considerably faster than
        looping and calling :py:meth:`ask` repeatedly.

        Args:
            image: PIL ``Image``, ``numpy`` array, or path to an image file.
            question: A question string *or* a list/tuple of question strings.
            word_boxes: Optional pre-extracted word-boxes in the LayoutLMv3
                format ``[[text, [x0, y0, x1, y1]], …]``.
            min_confidence: Minimum confidence threshold below which an answer
                will be marked as ``found = False``.
            debug: If ``True`` intermediate artefacts will be written to
                *debug_output_dir* to aid troubleshooting.
            debug_output_dir: Directory where debug artefacts should be saved.

        Returns:
            • A single :class:`QAResult` when *question* is a string.
            • A ``list`` of :class:`QAResult`` objects (one per question) when
              *question* is a list/tuple.
        """
        if not self._is_initialized:
            raise RuntimeError("DocumentQA is not properly initialized")

        # Normalise *questions* to a list so we can treat batch and single
        # uniformly.  We'll remember if the caller supplied a single question
        # so that we can preserve the original return type.
        single_question = False
        if isinstance(question, str):
            questions = [question]
            single_question = True
        elif isinstance(question, (list, tuple)) and all(isinstance(q, str) for q in question):
            questions = list(question)
        else:
            raise TypeError("'question' must be a string or a list/tuple of strings")

        # Process the image
        if isinstance(image, str):
            # It's a file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image_obj = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image_obj = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # Already a PIL Image
            image_obj = image
        else:
            raise TypeError("Image must be a PIL Image, numpy array, or file path")

        # ------------------------------------------------------------------
        # Build the queries for the pipeline (either single dict or list).
        # ------------------------------------------------------------------
        def _build_query_dict(q: str):
            d = {"image": image_obj, "question": q}
            if word_boxes:
                d["word_boxes"] = word_boxes
            return d

        queries = [_build_query_dict(q) for q in questions]

        # Save debug information if requested
        if debug:
            # Create debug directory
            os.makedirs(debug_output_dir, exist_ok=True)

            # Save the image
            image_debug_path = os.path.join(debug_output_dir, "debug_qa_image.png")
            image_obj.save(image_debug_path)

            # Save word boxes
            if word_boxes:
                word_boxes_path = os.path.join(debug_output_dir, "debug_qa_word_boxes.json")
                with open(word_boxes_path, "w") as f:
                    json.dump(word_boxes, f, indent=2)

                # Generate a visualization of the boxes on the image
                vis_image = image_obj.copy()
                draw = ImageDraw.Draw(vis_image)

                for i, (text, box) in enumerate(word_boxes):
                    x0, y0, x1, y1 = box
                    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
                    # Add text index for reference
                    draw.text((x0, y0), str(i), fill=(255, 0, 0))

                vis_path = os.path.join(debug_output_dir, "debug_qa_boxes_vis.png")
                vis_image.save(vis_path)

                logger.info(f"Saved debug files to {debug_output_dir}")
                logger.info(f"Question: {question}")
                logger.info(f"Image: {image_debug_path}")
                logger.info(f"Word boxes: {word_boxes_path}")
                logger.info(f"Visualization: {vis_path}")

        # ------------------------------------------------------------------
        # Run the queries through the pipeline (batch or single) and collect
        # *only the top answer* for each, mirroring the original behaviour.
        # ------------------------------------------------------------------
        logger.info(
            f"Running document QA pipeline with {len(queries)} question{'s' if len(queries) != 1 else ''}."
        )

        # When we pass a list the pipeline returns a list of per-question
        # results; each per-question result is itself a list (top-k answers).
        # We keep only the best answer (index 0) to maintain backwards
        # compatibility.
        raw_results = self.pipe(queries if len(queries) > 1 else queries[0])

        # Ensure we always have a list aligned with *questions*
        if len(queries) == 1:
            raw_results = [raw_results]

        processed_results: List[QAResult] = []

        for q, res in zip(questions, raw_results):
            top_res = res[0] if isinstance(res, list) else res  # pipeline may or may not nest

            # Save per-question result in debug mode
            if debug:
                # File names: debug_qa_result_0.json, …
                result_path = os.path.join(
                    debug_output_dir, f"debug_qa_result_{q[:30].replace(' ', '_')}.json"
                )
                try:
                    with open(result_path, "w") as f:
                        serializable = {
                            k: (
                                str(v)
                                if not isinstance(
                                    v, (str, int, float, bool, list, dict, type(None))
                                )
                                else v
                            )
                            for k, v in top_res.items()
                        }
                        json.dump(serializable, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save debug QA result for question '{q}': {e}")

            # Apply confidence threshold
            if top_res["score"] < min_confidence:
                qa_res = QAResult(
                    question=q,
                    answer="",
                    confidence=top_res["score"],
                    start=top_res.get("start", -1),
                    end=top_res.get("end", -1),
                    found=False,
                )
            else:
                qa_res = QAResult(
                    question=q,
                    answer=top_res["answer"],
                    confidence=top_res["score"],
                    start=top_res.get("start", 0),
                    end=top_res.get("end", 0),
                    found=True,
                )

            processed_results.append(qa_res)

        # Return appropriately typed result (single item or list)
        return processed_results[0] if single_question else processed_results

    def ask_pdf_page(
        self,
        page,
        question: Union[str, List[str], Tuple[str, ...]],
        min_confidence: float = 0.1,
        debug: bool = False,
    ) -> Union[QAResult, List[QAResult]]:
        """
        Ask a question about a specific PDF page.

        Args:
            page: natural_pdf.core.page.Page object
            question: Question to ask about the page
            min_confidence: Minimum confidence threshold for answers

        Returns:
            QAResult instance with answer details
        """
        # Ensure we have text elements on the page
        elements = page.find_all("text")
        if not elements:
            # Warn that no text was found and recommend OCR
            warnings.warn(
                f"No text elements found on page {page.index}. "
                "Consider applying OCR first using page.apply_ocr() to extract text from images.",
                UserWarning,
            )

            # Return appropriate "not found" result(s)
            if isinstance(question, (list, tuple)):
                return [
                    QAResult(
                        question=q,
                        answer="",
                        confidence=0.0,
                        start=-1,
                        end=-1,
                        found=False,
                    )
                    for q in question
                ]
            else:
                return QAResult(
                    question=question,
                    answer="",
                    confidence=0.0,
                    start=-1,
                    end=-1,
                    found=False,
                )

        # Extract word boxes
        word_boxes = self._get_word_boxes_from_elements(elements, offset_x=0, offset_y=0)

        # Generate a high-resolution image of the page
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        # Save a high resolution image (300 DPI)
        # Use render() for clean image without highlights
        page_image = page.render(resolution=300)
        page_image.save(temp_path)

        try:
            # Ask the question(s)
            result_obj = self.ask(
                image=temp_path,
                question=question,
                word_boxes=word_boxes,
                min_confidence=min_confidence,
                debug=debug,
            )

            # Ensure we have a list for uniform processing
            results = result_obj if isinstance(result_obj, list) else [result_obj]

            for res in results:
                # Attach page reference
                res.page_num = page.index

                # Map answer span back to source elements
                if res.found and "start" in res and "end" in res:
                    start_idx = res.start
                    end_idx = res.end

                    if (
                        elements
                        and 0 <= start_idx < len(word_boxes)
                        and 0 <= end_idx < len(word_boxes)
                    ):
                        matched_texts = [wb[0] for wb in word_boxes[start_idx : end_idx + 1]]

                        source_elements = []
                        for element in elements:
                            if hasattr(element, "text") and element.text in matched_texts:
                                source_elements.append(element)
                                if element.text in matched_texts:
                                    matched_texts.remove(element.text)

                        res.source_elements = ElementCollection(source_elements)

            # Return result(s) preserving original input type
            if isinstance(question, (list, tuple)):
                return results
            else:
                return results[0]

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def ask_pdf_region(
        self,
        region,
        question: Union[str, List[str], Tuple[str, ...]],
        min_confidence: float = 0.1,
        debug: bool = False,
    ) -> Union[QAResult, List[QAResult]]:
        """
        Ask a question about a specific region of a PDF page.

        Args:
            region: natural_pdf.elements.region.Region object
            question: Question to ask about the region
            min_confidence: Minimum confidence threshold for answers

        Returns:
            QAResult instance with answer details
        """
        # Get all text elements within the region
        elements = region.find_all("text")

        # Check if we have text elements
        if not elements:
            # Warn that no text was found and recommend OCR
            warnings.warn(
                f"No text elements found in region on page {region.page.index}. "
                "Consider applying OCR first using region.apply_ocr() to extract text from images.",
                UserWarning,
            )

            # Return appropriate "not found" result(s)
            if isinstance(question, (list, tuple)):
                return [
                    QAResult(
                        question=q,
                        answer="",
                        confidence=0.0,
                        start=-1,
                        end=-1,
                        found=False,
                    )
                    for q in question
                ]
            else:
                return QAResult(
                    question=question,
                    answer="",
                    confidence=0.0,
                    start=-1,
                    end=-1,
                    found=False,
                )

        # Extract word boxes adjusted for the cropped region
        x0, top = int(region.x0), int(region.top)
        word_boxes = self._get_word_boxes_from_elements(elements, offset_x=x0, offset_y=top)

        # Generate a cropped image of the region
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        # Get page image at high resolution - this returns a PIL Image directly
        # Use render() for clean image without highlights
        page_image = region.page.render(resolution=300)

        # Crop to region
        x0, top, x1, bottom = int(region.x0), int(region.top), int(region.x1), int(region.bottom)
        region_image = page_image.crop((x0, top, x1, bottom))
        region_image.save(temp_path)

        try:
            # Ask the question(s)
            result_obj = self.ask(
                image=temp_path,
                question=question,
                word_boxes=word_boxes,
                min_confidence=min_confidence,
                debug=debug,
            )

            results = result_obj if isinstance(result_obj, list) else [result_obj]

            for res in results:
                res.region = region
                res.page_num = region.page.index

                if res.found and "start" in res and "end" in res:
                    start_idx = res.start
                    end_idx = res.end

                    if (
                        elements
                        and 0 <= start_idx < len(word_boxes)
                        and 0 <= end_idx < len(word_boxes)
                    ):
                        matched_texts = [wb[0] for wb in word_boxes[start_idx : end_idx + 1]]

                        source_elements = []
                        for element in elements:
                            if hasattr(element, "text") and element.text in matched_texts:
                                source_elements.append(element)
                                if element.text in matched_texts:
                                    matched_texts.remove(element.text)

                        res.source_elements = ElementCollection(source_elements)

            return results if isinstance(question, (list, tuple)) else results[0]

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
