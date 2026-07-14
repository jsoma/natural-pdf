import json
import logging
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image, ImageDraw

from natural_pdf.utils.optional_imports import require

logger = logging.getLogger("natural_pdf.qa.document_qa")


_qa_engine_cache: Dict[str, "DocumentQA"] = {}


def get_qa_engine(model_name: str = "impira/layoutlm-document-qa", **kwargs):
    """Get a cached DocumentQA engine instance.

    Args:
        model_name: Name of the model to use (default: "impira/layoutlm-document-qa")
        **kwargs: Additional parameters to pass to the DocumentQA constructor

    Returns:
        DocumentQA instance
    """
    if model_name not in _qa_engine_cache:
        _qa_engine_cache[model_name] = DocumentQA(model_name=model_name, **kwargs)
    return _qa_engine_cache[model_name]


class DocumentQA:
    """
    Document Question Answering using LayoutLM.

    This class provides the ability to ask natural language questions about document content,
    leveraging the spatial layout information from PDF pages.
    """

    def __init__(
        self,
        model_name: str = "impira/layoutlm-document-qa",
        device: Optional[str] = None,
    ):
        """
        Initialize the Document QA engine.

        Args:
            model_name: HuggingFace model name to use (default: "impira/layoutlm-document-qa")
            device: Device to run the model on ('cuda' or 'cpu'). If None, will use cuda if available.
        """
        try:
            torch = require("torch")
            transformers_mod = require("transformers")
            pipeline = getattr(transformers_mod, "pipeline")
        except ImportError as exc:
            self._is_initialized = False
            raise ImportError(
                "DocumentQA requires torch and transformers. Install with: pip install torch transformers"
            ) from exc

        logger.info(f"Initializing DocumentQA with model {model_name} on {device}")

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if device is None and torch.backends.mps.is_available():
            try:
                self.pipe = pipeline("document-question-answering", model=model_name, device="mps")
                self.device = "mps"
            except RuntimeError as e:
                logger.warning(f"MPS OOM: {e}, falling back to CPU")
                self.pipe = pipeline("document-question-answering", model=model_name, device="cpu")
                self.device = "cpu"
        else:
            self.pipe = pipeline(
                "document-question-answering", model=model_name, device=resolved_device
            )
            self.device = resolved_device

        self.model_name = model_name
        self._is_initialized = True

    def is_available(self) -> bool:
        """Check if the QA engine is properly initialized."""
        return self._is_initialized

    def _get_word_boxes_from_elements(
        self, elements: Iterable[Any], offset_x: float = 0, offset_y: float = 0
    ) -> List[List[Any]]:
        """
        Extract source-coordinate word boxes from text elements.

        The returned boxes retain the caller's logical coordinate system.
        Callers rendering PDF content should normalize them against the page
        or region bounds before passing them to a LayoutLM-family model.

        Args:
            elements: List of TextElement objects
            offset_x: X-coordinate offset to subtract from each box.
            offset_y: Y-coordinate offset to subtract from each box.

        Returns:
            List of ``[text, [x0, top, x1, bottom]]`` entries in the source
            document's coordinate system.
        """
        word_boxes = []

        for element in elements:
            if hasattr(element, "text") and element.text.strip():
                word_boxes.append(
                    [
                        element.text,
                        [
                            float(element.x0) - offset_x,
                            float(element.top) - offset_y,
                            float(element.x1) - offset_x,
                            float(element.bottom) - offset_y,
                        ],
                    ]
                )

        return word_boxes

    @staticmethod
    def _normalize_word_boxes(
        word_boxes: List[List[Any]], bounds: Tuple[float, float, float, float]
    ) -> List[List[Any]]:
        """Map source-coordinate boxes into Transformers' ``0..1000`` space.

        LayoutLM tokenizers expect every bounding box to be relative to the
        image supplied to the pipeline, normalized independently on each axis
        to an inclusive ``0..1000`` range.  In particular, PDF coordinates are
        points while a 300-DPI render is pixels, so passing the raw PDF values
        causes text and image geometry to disagree.
        """
        left, top, right, bottom = (float(value) for value in bounds)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            raise ValueError("word-box bounds must have positive width and height")

        def _normalize(value: Any, origin: float, size: float) -> int:
            # Clamping before conversion handles elements that only partly
            # overlap a Region while keeping the tokenizer contract intact.
            return max(0, min(1000, int(round((float(value) - origin) * 1000 / size))))

        normalized: List[List[Any]] = []
        for entry in word_boxes:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("word_boxes entries must be [text, [x0, y0, x1, y1]]")
            text, box = entry
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("word_boxes entries must contain four box coordinates")

            x0 = _normalize(box[0], left, width)
            y0 = _normalize(box[1], top, height)
            x1 = _normalize(box[2], left, width)
            y1 = _normalize(box[3], top, height)
            normalized.append([text, [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]])

        return normalized

    def ask(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: Union[str, List[str], Tuple[str, ...]],
        word_boxes: Optional[List[List[Any]]] = None,
        min_confidence: float = 0.1,
        debug: bool = False,
        debug_output_dir: str = "output",
        *,
        handle_impossible_answer: bool = True,
        max_answer_len: int = 30,
    ) -> Union[dict, List[dict]]:
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
            word_boxes: Optional pre-extracted word boxes in Transformers'
                normalized ``0..1000`` LayoutLM coordinate system.
            min_confidence: Minimum confidence threshold below which an answer
                will be marked as ``found = False``.
            handle_impossible_answer: Include the model's no-answer candidate.
                Defaults to ``True`` so absent answers are not forced into a
                plausible-looking source span.
            max_answer_len: Maximum answer length in tokens.  Defaults to 30,
                which keeps extractive answers concise while allowing typical
                document values and labels.
            debug: If ``True`` intermediate artefacts will be written to
                *debug_output_dir* to aid troubleshooting.
            debug_output_dir: Directory where debug artefacts should be saved.

        Returns:
            • A single :class:`dict` when *question* is a string.
            • A ``list`` of :class:`dict`` objects (one per question) when
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

        if not questions:
            return []

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
                    x0 = round(x0 * image_obj.width / 1000)
                    y0 = round(y0 * image_obj.height / 1000)
                    x1 = round(x1 * image_obj.width / 1000)
                    y1 = round(y1 * image_obj.height / 1000)
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
        pipeline_output = self.pipe(
            queries if len(queries) > 1 else queries[0],
            handle_impossible_answer=handle_impossible_answer,
            max_answer_len=max_answer_len,
        )

        if len(queries) == 1:
            normalized_output = [pipeline_output]
        else:
            normalized_output = pipeline_output

        raw_results = cast(
            List[Union[Dict[str, Any], List[Dict[str, Any]]]],
            normalized_output,
        )
        if len(raw_results) < len(questions):
            raw_results = [*raw_results, *([{}] * (len(questions) - len(raw_results)))]

        processed_results: List[dict] = []

        for q, res in zip(questions, raw_results):
            # A no-answer candidate is normally returned as a dictionary with
            # an empty ``answer``.  Some pipeline/model combinations instead
            # return an empty list, so normalize that case as well.
            top_res = res[0] if isinstance(res, list) and res else (res or {})

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
                except (OSError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to save debug QA result for question '{q}': {e}")

            # Apply confidence threshold
            score = top_res.get("score", 0.0)
            answer = top_res.get("answer", "")
            if score < min_confidence or not answer:
                qa_res = dict(
                    question=q,
                    answer="",
                    confidence=score,
                    start=-1 if not answer else top_res.get("start", -1),
                    end=-1 if not answer else top_res.get("end", -1),
                    found=False,
                )
            else:
                qa_res = dict(
                    question=q,
                    answer=answer,
                    confidence=score,
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
        *,
        handle_impossible_answer: bool = True,
        max_answer_len: int = 30,
    ) -> Union[dict, List[dict]]:
        """
        Ask a question about a specific PDF page.

        Args:
            page: natural_pdf.core.page.Page object
            question: Question to ask about the page
            min_confidence: Minimum confidence threshold for answers
            handle_impossible_answer: Include no-answer candidates from the
                model. Defaults to ``True``.
            max_answer_len: Maximum answer length in tokens. Defaults to 30.

        Returns:
            dict instance with answer details
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
                    dict(
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
                return dict(
                    question=question,
                    answer="",
                    confidence=0.0,
                    start=-1,
                    end=-1,
                    found=False,
                )

        # Extract word boxes
        word_box_elements = [
            element for element in elements if hasattr(element, "text") and element.text.strip()
        ]
        word_boxes = self._normalize_word_boxes(
            self._get_word_boxes_from_elements(word_box_elements),
            (0.0, 0.0, float(page.width), float(page.height)),
        )

        # Pass the clean high-resolution render directly.  The word boxes are
        # normalized from PDF points relative to the page bounds above.
        page_image = page.render(resolution=300, highlights=False)

        # Ask the question(s)
        result_obj = self.ask(
            image=page_image,
            question=question,
            word_boxes=word_boxes,
            min_confidence=min_confidence,
            handle_impossible_answer=handle_impossible_answer,
            max_answer_len=max_answer_len,
            debug=debug,
        )

        # Ensure we have a list for uniform processing
        results = result_obj if isinstance(result_obj, list) else [result_obj]

        for res in results:
            # Attach page reference
            res["page_num"] = page.index

            # Pipeline span indices refer to the filtered word-box sequence,
            # not all page elements.  Slice that exact sequence so repeated
            # text values cannot map to an earlier, unrelated occurrence.
            if res.get("found") and "start" in res and "end" in res:
                start_idx = res["start"]
                end_idx = res["end"]

                if 0 <= start_idx <= end_idx < len(word_box_elements):
                    from natural_pdf.elements.element_collection import ElementCollection

                    res["source_elements"] = ElementCollection(
                        word_box_elements[start_idx : end_idx + 1]
                    )

        # Return result(s) preserving original input type
        return results if isinstance(question, (list, tuple)) else results[0]

    def ask_pdf_region(
        self,
        region,
        question: Union[str, List[str], Tuple[str, ...]],
        min_confidence: float = 0.1,
        debug: bool = False,
        *,
        handle_impossible_answer: bool = True,
        max_answer_len: int = 30,
    ) -> Union[dict, List[dict]]:
        """
        Ask a question about a specific region of a PDF page.

        Args:
            region: natural_pdf.elements.region.Region object
            question: Question to ask about the region
            min_confidence: Minimum confidence threshold for answers
            handle_impossible_answer: Include no-answer candidates from the
                model. Defaults to ``True``.
            max_answer_len: Maximum answer length in tokens. Defaults to 30.

        Returns:
            dict instance with answer details
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
                    dict(
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
                return dict(
                    question=question,
                    answer="",
                    confidence=0.0,
                    start=-1,
                    end=-1,
                    found=False,
                )

        word_box_elements = [
            element for element in elements if hasattr(element, "text") and element.text.strip()
        ]
        region_bounds = (
            float(region.x0),
            float(region.top),
            float(region.x1),
            float(region.bottom),
        )
        word_boxes = self._normalize_word_boxes(
            self._get_word_boxes_from_elements(word_box_elements), region_bounds
        )

        # Let Region render its own crop.  It knows how to map the PDF-space
        # region bounds to the requested render resolution, avoiding a raw
        # PDF-point crop against a 300-DPI page image.
        region_image = region.render(resolution=300, crop=True, highlights=False)

        # Ask the question(s)
        result_obj = self.ask(
            image=region_image,
            question=question,
            word_boxes=word_boxes,
            min_confidence=min_confidence,
            handle_impossible_answer=handle_impossible_answer,
            max_answer_len=max_answer_len,
            debug=debug,
        )

        results = result_obj if isinstance(result_obj, list) else [result_obj]

        for res in results:
            res["region"] = region
            res["page_num"] = region.page.index

            if res.get("found") and "start" in res and "end" in res:
                start_idx = res["start"]
                end_idx = res["end"]

                if 0 <= start_idx <= end_idx < len(word_box_elements):
                    from natural_pdf.elements.element_collection import ElementCollection

                    res["source_elements"] = ElementCollection(
                        word_box_elements[start_idx : end_idx + 1]
                    )

        return results if isinstance(question, (list, tuple)) else results[0]
