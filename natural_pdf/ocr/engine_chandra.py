# engine_chandra.py
import importlib.util
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, ChandraOCROptions

logger = logging.getLogger(__name__)


class ChandraOCREngine(OCREngine):
    """Chandra OCR engine — VLM-based successor to Surya.

    Chandra uses a single vision-language model (based on Qwen) to perform
    OCR with layout detection in one pass. It returns block-level bounding
    boxes with HTML/markdown text.

    Install: ``pip install chandra-ocr[hf]``
    """

    def __init__(self):
        super().__init__()
        self._manager = None

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the Chandra InferenceManager."""
        if not self.is_available():
            raise ImportError(
                "Chandra OCR is not installed. Install with: pip install chandra-ocr[hf]"
            )

        from chandra.model import InferenceManager  # type: ignore[import-untyped]

        opts = options if isinstance(options, ChandraOCROptions) else ChandraOCROptions()

        method = opts.method
        kwargs: Dict[str, Any] = {}
        if method == "vllm" and opts.vllm_url:
            kwargs["url"] = opts.vllm_url

        self.logger.info("Initializing Chandra InferenceManager (method=%s)...", method)
        self._manager = InferenceManager(method=method, **kwargs)
        self.logger.info("Chandra InferenceManager ready.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Chandra works with PIL images directly."""
        return image.convert("RGB") if image.mode != "RGB" else image

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Run Chandra OCR on a single image."""
        if self._manager is None:
            raise RuntimeError("Chandra InferenceManager is not initialized.")

        from chandra.model import BatchInputItem  # type: ignore[import-untyped]

        opts = options if isinstance(options, ChandraOCROptions) else ChandraOCROptions()

        prompt_type = "ocr_layout" if not detect_only else "ocr_layout"
        max_tokens = opts.max_output_tokens

        batch = [BatchInputItem(image=image, prompt_type=prompt_type)]
        outputs = self._manager.generate(batch, max_output_tokens=max_tokens)
        return outputs[0] if outputs else None

    # Regex to strip HTML tags for plain text extraction from chunk content
    _RE_HTML_TAGS = re.compile(r"<[^>]+>")
    _RE_MULTI_SPACE = re.compile(r"  +")
    _RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

    @classmethod
    def _html_to_text(cls, html: str) -> str:
        """Convert HTML fragment to plain text."""
        text = cls._RE_HTML_TAGS.sub("", html)
        text = cls._RE_MULTI_SPACE.sub(" ", text)
        text = cls._RE_MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool, **kwargs
    ) -> List[TextRegion]:
        """Convert Chandra output to TextRegion objects."""
        if raw_results is None:
            return []

        if getattr(raw_results, "error", False):
            self.logger.warning("Chandra returned an error for this image.")
            return []

        regions: List[TextRegion] = []

        # Use chunks for block-level results with bboxes
        chunks = getattr(raw_results, "chunks", None)
        if chunks and isinstance(chunks, list):
            for chunk in chunks:
                bbox_raw = chunk.get("bbox")
                if not bbox_raw or len(bbox_raw) != 4:
                    continue

                bbox: Tuple[float, float, float, float] = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )

                if detect_only:
                    regions.append(TextRegion(bbox, "", 0.0, source="chandra"))
                else:
                    content = chunk.get("content", "")
                    text = self._html_to_text(content)
                    if text:
                        # Chandra doesn't provide confidence scores; use 1.0
                        regions.append(TextRegion(bbox, text, 1.0, source="chandra"))

        elif not detect_only:
            # Fallback: no chunks available, use full markdown as a single region
            markdown = getattr(raw_results, "markdown", "")
            if markdown:
                page_box = getattr(raw_results, "page_box", None)
                if page_box and len(page_box) == 4:
                    bbox = (
                        float(page_box[0]),
                        float(page_box[1]),
                        float(page_box[2]),
                        float(page_box[3]),
                    )
                else:
                    bbox = (0.0, 0.0, 1.0, 1.0)
                regions.append(TextRegion(bbox, markdown, 1.0, source="chandra"))

        return regions

    def is_available(self) -> bool:
        """Check if the chandra library is installed."""
        return importlib.util.find_spec("chandra") is not None
