# layout_detector_paddle.py
import logging
import importlib.util
import os
import tempfile
from typing import List, Dict, Any, Optional
from PIL import Image

# Assuming base class and options are importable
try:
    from .base import LayoutDetector
    from .layout_options import PaddleLayoutOptions, BaseLayoutOptions
except ImportError:
    # Placeholders if run standalone or imports fail
    class BaseLayoutOptions: pass
    class PaddleLayoutOptions(BaseLayoutOptions): pass
    class LayoutDetector:
         def __init__(self): self.logger=logging.getLogger(); self.supported_classes=set()
         def _get_model(self, options): raise NotImplementedError
         def _normalize_class_name(self, n): return n
         def validate_classes(self, c): pass
    logging.basicConfig()

logger = logging.getLogger(__name__)

# Check for dependencies
paddle_spec = importlib.util.find_spec("paddle") or importlib.util.find_spec("paddlepaddle")
paddleocr_spec = importlib.util.find_spec("paddleocr")
PPStructure = None
PaddleOCR = None # For optional text detection

if paddle_spec and paddleocr_spec:
    try:
        from paddleocr import PPStructure, PaddleOCR
    except ImportError as e:
        logger.warning(f"Could not import Paddle dependencies: {e}")
else:
    logger.warning("paddlepaddle or paddleocr not found. PaddleLayoutDetector will not be available.")


class PaddleLayoutDetector(LayoutDetector):
    """Document layout and table structure detector using PaddlePaddle's PP-Structure."""

    def __init__(self):
        super().__init__()
        # Supported classes by PP-Structure (adjust based on model version/capabilities)
        self.supported_classes = {
            'text', 'title', 'figure', 'figure_caption',
            'table', 'table_caption', 'table_cell', # Added table_cell
            'header', 'footer', 'reference', 'equation',
            # PP-StructureV2 might add others like list, pub_number etc.
        }
        # Models are loaded via _get_model

    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        return PPStructure is not None and PaddleOCR is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """Generate cache key based on language and device."""
        if not isinstance(options, PaddleLayoutOptions):
            options = PaddleLayoutOptions(device=options.device) # Use base device

        device_key = str(options.device).lower() if options.device else 'default_device'
        lang_key = options.lang
        # Key could also include enable_table, use_angle_cls if these affect model loading fundamentally
        # For PPStructure, they are primarily runtime flags, so lang/device might suffice for caching the *instance*.
        return f"{self.__class__.__name__}_{device_key}_{lang_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """Load the PPStructure model based on options."""
        if not self.is_available():
             raise RuntimeError("Paddle dependencies (paddlepaddle, paddleocr) not installed.")

        if not isinstance(options, PaddleLayoutOptions):
            raise TypeError("Incorrect options type provided for Paddle model loading.")

        self.logger.info(f"Loading PPStructure model (lang={options.lang}, device={options.device}, table={options.enable_table})...")
        try:
            # PPStructure init takes several arguments that control runtime behavior
            # We cache the instance based on lang/device, assuming other flags don't require reloading.
            # Note: show_log is a runtime arg, not needed for instance caching key.
            # Note: `layout=False` disables layout analysis, which we definitely want here.
            # Note: `ocr=False` might disable text detection needed for table structure? Check PPStructure docs.
            # It seems best to initialize with core settings and pass others during the call if possible.
            # However, PPStructure call signature is simple (__call__(self, img, ...))
            # So, we likely need to initialize with most settings.
            model_instance = PPStructure(
                 lang=options.lang,
                 use_gpu=('cuda' in str(options.device).lower() or 'gpu' in str(options.device).lower()),
                 use_angle_cls=options.use_angle_cls,
                 show_log=options.show_log,
                 layout=True, # Ensure layout analysis is on
                 table=options.enable_table, # Control table analysis
                 ocr=False # Usually disable internal OCR if only using for layout/table
                 # Add other PPStructure init args from options.extra_args if needed
                 # **options.extra_args
            )
            self.logger.info("PPStructure model loaded.")
            return model_instance
        except Exception as e:
            self.logger.error(f"Failed to load PPStructure model: {e}", exc_info=True)
            raise

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect layout elements in an image using PaddlePaddle."""
        if not self.is_available():
            raise RuntimeError("Paddle dependencies (paddlepaddle, paddleocr) not installed.")

        if not isinstance(options, PaddleLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected PaddleLayoutOptions. Using defaults.")
             options = PaddleLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args
                 # Other Paddle options will use defaults
             )

        self.validate_classes(options.classes or [])
        if options.exclude_classes:
            self.validate_classes(options.exclude_classes)

        # Get the cached/loaded PPStructure instance
        ppstructure_instance = self._get_model(options)

        # PPStructure call requires an image path. Save temp file.
        detections = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = os.path.join(temp_dir, f"paddle_input_{os.getpid()}.png")
            try:
                self.logger.debug(f"Saving temporary image for Paddle detector to: {temp_image_path}")
                image.convert("RGB").save(temp_image_path) # Ensure RGB

                # Process image with PP-Structure instance
                # The instance was configured during _load_model_from_options
                self.logger.debug("Running PPStructure analysis...")
                result = ppstructure_instance(temp_image_path)
                self.logger.debug(f"PPStructure returned {len(result)} regions.")

            except Exception as e:
                self.logger.error(f"Error during PPStructure analysis: {e}", exc_info=True)
                # Clean up temp file before raising or returning
                if os.path.exists(temp_image_path):
                    try: os.remove(temp_image_path)
                    except OSError as e_rm: self.logger.warning(f"Could not remove temp file {temp_image_path}: {e_rm}")
                raise # Re-raise error

            finally:
                 # Ensure cleanup even if analysis worked
                 if os.path.exists(temp_image_path):
                      try: os.remove(temp_image_path)
                      except OSError as e_rm: self.logger.warning(f"Could not remove temp file {temp_image_path}: {e_rm}")

        # --- Process Results ---
        if not result:
            self.logger.warning("PaddleLayout returned empty results")
            return []

        # Prepare normalized class filters once
        normalized_classes_req = {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        normalized_classes_excl = {self._normalize_class_name(c) for c in options.exclude_classes} if options.exclude_classes else set()

        for region in result:
            try:
                region_type_orig = region.get('type', 'unknown')
                # Handle potential list returns for type (seen in some versions)
                if isinstance(region_type_orig, list):
                     region_type_orig = region_type_orig[0] if region_type_orig else 'unknown'

                region_type = region_type_orig.lower()
                normalized_class = self._normalize_class_name(region_type)

                # Apply class filtering
                if normalized_classes_req and normalized_class not in normalized_classes_req: continue
                if normalized_class in normalized_classes_excl: continue

                # PP-Structure results don't always have confidence, use threshold or default
                confidence_score = region.get('score', 1.0) # Default to 1.0 if missing
                if confidence_score < options.confidence: continue

                bbox = region.get('bbox')
                if not bbox or len(bbox) != 4:
                     self.logger.warning(f"Skipping region with invalid bbox: {region}")
                     continue
                x_min, y_min, x_max, y_max = map(float, bbox)

                # Add detection
                detection_data = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'class': region_type_orig, # Keep original case if needed
                    'confidence': confidence_score,
                    'normalized_class': normalized_class,
                    'source': 'layout',
                    'model': 'paddle'
                }
                detections.append(detection_data)

                # --- Process Table Cells (if enabled and present) ---
                if region_type == 'table' and options.enable_table and 'res' in region:
                    process_cells = (normalized_classes_req is None or 'table-cell' in normalized_classes_req) and \
                                    ('table-cell' not in normalized_classes_excl)

                    if process_cells and isinstance(region['res'], list): # V2 structure
                         for cell in region['res']:
                              if 'box' not in cell or len(cell['box']) != 4: continue
                              cell_bbox = cell['box']
                              cell_x_min, cell_y_min, cell_x_max, cell_y_max = map(float, cell_bbox)
                              # Add cell detection (confidence often not available per cell)
                              detections.append({
                                  'bbox': (cell_x_min, cell_y_min, cell_x_max, cell_y_max),
                                  'class': 'table cell', # Standardize name
                                  'confidence': confidence_score * 0.95, # Inherit table confidence (slightly reduced)
                                  'normalized_class': 'table-cell',
                                  'text': cell.get('text', ''), # Include text if available
                                  'source': 'layout', 'model': 'paddle'
                              })
                    elif process_cells and isinstance(region['res'], dict) and 'cells' in region['res']: # Older structure
                         # Handle older 'cells' list if needed (logic from original file)
                         pass # Add logic based on original paddle.txt if supporting older PP-Structure

            except (TypeError, KeyError, IndexError, ValueError) as e:
                self.logger.warning(f"Error processing Paddle region: {region}. Error: {e}")
                continue

        # --- Optional: Add Text Boxes from separate OCR run ---
        if options.detect_text:
             # This requires another model instance (PaddleOCR) and adds complexity.
             # Consider if this is truly needed or if layout regions are sufficient.
             # If needed, implement similar to original paddle.txt:
             # - Instantiate PaddleOCR (potentially cache separately)
             # - Run ocr(img_path, det=True, rec=False)
             # - Process results, adding 'text' class detections
             self.logger.info("Paddle detect_text=True: Running separate OCR text detection...")
             # (Implementation omitted for brevity - requires PaddleOCR instance)
             pass


        self.logger.info(f"PaddleLayout detected {len(detections)} layout elements matching criteria.")
        return detections

