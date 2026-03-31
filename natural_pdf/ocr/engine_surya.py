# ocr_engine_surya.py
import importlib.util
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

from PIL import Image

logger = logging.getLogger(__name__)

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, SuryaOCROptions


def _surya_compat_hint() -> str:
    """Build a version-aware compatibility hint."""
    tf_ver = ""
    try:
        import transformers

        tf_ver = getattr(transformers, "__version__", "")
    except ImportError:
        pass

    if tf_ver and tf_ver.startswith("5"):
        return (
            f"surya-ocr is not yet compatible with transformers {tf_ver}. "
            "Surya requires transformers 4.x. If other dependencies need transformers 5, "
            "you may need to use a different OCR engine (easyocr, paddlevl, or vlm) "
            "until surya releases a compatible update.\n"
            'To downgrade: pip install "transformers<5"'
        )
    return (
        "Your version of surya-ocr is not compatible with your installed transformers. "
        "Try: pip install --upgrade surya-ocr transformers\n"
        "See: https://github.com/datalab-to/surya/issues/484"
    )


class SuryaOCREngine(OCREngine):
    """Surya OCR engine implementation."""

    def __init__(self):
        super().__init__()
        self._recognition_predictor: Optional[Callable[..., Any]] = None
        self._detection_predictor: Optional[Callable[..., Any]] = None
        self._surya_recognition: Optional[Callable[..., Any]] = None
        self._surya_detection: Optional[Callable[..., Any]] = None
        self._langs: Sequence[str] = self.DEFAULT_LANGUAGES

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize Surya predictors."""
        if not self.is_available():
            raise ImportError(
                "Surya OCR library is not installed. Install with: pip install surya-ocr"
            )

        self._langs = languages or self.DEFAULT_LANGUAGES

        from surya.detection import DetectionPredictor  # type: ignore[import-untyped]
        from surya.recognition import RecognitionPredictor  # type: ignore[import-untyped]

        self._surya_recognition = RecognitionPredictor
        self._surya_detection = DetectionPredictor
        self.logger.info("Surya modules imported successfully.")

        try:
            self.logger.info("Instantiating Surya DetectionPredictor (device=%s)...", device)
            self._detection_predictor = DetectionPredictor(device=device)

            self.logger.info("Instantiating Surya RecognitionPredictor (device=%s)...", device)
            # Surya >= 0.17: RecognitionPredictor requires a FoundationPredictor
            # Surya < 0.17: RecognitionPredictor takes no required args
            try:
                from surya.foundation import FoundationPredictor  # type: ignore[import-untyped]

                foundation = FoundationPredictor(device=device)
                self._recognition_predictor = RecognitionPredictor(foundation)
            except ImportError:
                # Older surya without FoundationPredictor
                self._recognition_predictor = RecognitionPredictor()
        except TypeError as exc:
            # Handle case where API changed in unexpected ways
            raise RuntimeError(
                f"Failed to initialize Surya predictors: {exc}\n{_surya_compat_hint()}"
            ) from exc
        except AttributeError as exc:
            if "pad_token_id" in str(exc) or "rope" in str(exc).lower():
                raise RuntimeError(_surya_compat_hint()) from exc
            raise
        except KeyError as exc:
            if "rope" in str(exc).lower() or "default" in str(exc).lower():
                raise RuntimeError(_surya_compat_hint()) from exc
            raise

        self.logger.info("Surya predictors initialized.")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Surya uses PIL images directly, so just return the image."""
        return image

    def _process_single_image(
        self, image: Any, detect_only: bool, options: Optional[BaseOCROptions]
    ) -> Any:
        """Process a single image with Surya OCR."""
        if self._recognition_predictor is None or self._detection_predictor is None:
            raise RuntimeError("Surya predictors are not initialized.")

        if not isinstance(image, Image.Image):
            raise TypeError("SuryaOCREngine expects PIL images after preprocessing")

        langs = [list(self._langs)]

        try:
            # Surya expects lists of images, so we need to wrap our single image
            if detect_only:
                detection_predictor = self._detection_predictor
                assert detection_predictor is not None
                results = detection_predictor(images=[image])
            else:
                # Some Surya versions require 'langs' parameter in the __call__ while
                # others assume the predictor was initialized with languages already.
                # Inspect the callable signature to decide what to pass.
                import inspect

                recog_callable = self._recognition_predictor
                try:
                    sig = inspect.signature(recog_callable)
                    has_langs_param = "langs" in sig.parameters
                except (TypeError, ValueError):
                    # Fallback: assume langs not required if signature cannot be inspected
                    has_langs_param = False

                recognition_predictor = self._recognition_predictor
                detection_predictor = self._detection_predictor
                assert recognition_predictor is not None
                assert detection_predictor is not None

                if has_langs_param:
                    results = recognition_predictor(
                        langs=langs,
                        images=[image],
                        det_predictor=detection_predictor,
                    )
                else:
                    # Older/newer Surya versions that omit 'langs'
                    results = recognition_predictor(
                        images=[image],
                        det_predictor=detection_predictor,
                    )
        except AttributeError as exc:
            if "pad_token_id" in str(exc) or "rope" in str(exc).lower():
                raise RuntimeError(_surya_compat_hint()) from exc
            raise
        except KeyError as exc:
            if "rope" in str(exc).lower() or "default" in str(exc).lower():
                raise RuntimeError(_surya_compat_hint()) from exc
            raise

        # Surya may return a list with one result per image or a single result object
        # Return the result as-is and handle the extraction in _standardize_results
        return results

    _RE_MATH = re.compile(r"<math>.*?</math>", re.DOTALL)
    _RE_BOLD = re.compile(r"</?b>")
    _RE_MULTI_SPACE = re.compile(r"  +")

    @classmethod
    def _clean_text(cls, text: str, strip_math: bool = True) -> str:
        """Strip Surya markup from OCR text.

        ``<b>``/``</b>`` tags are always removed.
        ``<math>…</math>`` blocks are removed when *strip_math* is True (the default).
        """
        if strip_math:
            text = cls._RE_MATH.sub("", text)
        text = cls._RE_BOLD.sub("", text)
        text = cls._RE_MULTI_SPACE.sub(" ", text)
        return text.strip()

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool, **kwargs
    ) -> List[TextRegion]:
        """Convert Surya results to standardized TextRegion objects."""
        opts = kwargs.get("options")
        strip_math = getattr(opts, "strip_math", True)

        standardized_regions: List[TextRegion] = []

        raw_result: Any
        if isinstance(raw_results, list) and raw_results:
            raw_result = raw_results[0]
        else:
            raw_result = raw_results

        if raw_result is None:
            return standardized_regions

        if not detect_only and hasattr(raw_result, "text_lines"):
            results_iter = getattr(raw_result, "text_lines", None)
        else:
            results_iter = getattr(raw_result, "bboxes", None)

        if results_iter is None:
            return standardized_regions

        if isinstance(results_iter, Iterable):
            results_iterable = results_iter
        else:
            results_iterable = [results_iter]

        for line in results_iterable:
            bbox_raw: Any = None
            try:
                bbox_raw = getattr(line, "bbox", None)
                if bbox_raw is None:
                    bbox_raw = getattr(line, "polygon", None)
                if bbox_raw is None:
                    raise ValueError("Missing bbox/polygon data")
                bbox = self._standardize_bbox(bbox_raw)
            except ValueError as e:
                raise ValueError(
                    f"Could not standardize bounding box from Surya result: {bbox_raw}"
                ) from e

            if detect_only:
                standardized_regions.append(TextRegion(bbox, "", 0.0))
            else:
                text = self._clean_text(getattr(line, "text", ""), strip_math=strip_math)
                confidence = float(getattr(line, "confidence", 0.0))
                if confidence >= min_confidence and text:
                    standardized_regions.append(TextRegion(bbox, text, confidence))

        return standardized_regions

    def is_available(self) -> bool:
        """Check if the surya library is installed."""
        return importlib.util.find_spec("surya") is not None
