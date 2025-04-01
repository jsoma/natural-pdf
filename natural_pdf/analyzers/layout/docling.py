# layout_detector_docling.py
import logging
import importlib.util
import os
import tempfile
from typing import List, Dict, Any, Optional
from PIL import Image

# Assuming base class and options are importable
try:
    from .base import LayoutDetector
    from .layout_options import DoclingLayoutOptions, BaseLayoutOptions
except ImportError:
    # Placeholders if run standalone or imports fail
    class BaseLayoutOptions: pass
    class DoclingLayoutOptions(BaseLayoutOptions): pass
    class LayoutDetector:
         def __init__(self): self.logger=logging.getLogger(); self.supported_classes=set()
         def _get_model(self, options): raise NotImplementedError
         def _normalize_class_name(self, n): return n
         def validate_classes(self, c): pass
    logging.basicConfig()

logger = logging.getLogger(__name__)

# Check for dependency
docling_spec = importlib.util.find_spec("docling")
DocumentConverter = None
if docling_spec:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        logger.warning(f"Could not import Docling dependencies: {e}")
else:
    logger.warning("docling not found. DoclingLayoutDetector will not be available.")


class DoclingLayoutDetector(LayoutDetector):
    """Document layout and text recognition using Docling."""

    def __init__(self):
        super().__init__()
        # Docling classes are dynamic/hierarchical, define common ones
        self.supported_classes = {
            'Header', 'Footer', 'Paragraph', 'Heading', 'List', 'ListItem',
            'Table', 'Figure', 'Caption', 'Footnote', 'PageNumber', 'Equation',
            'Code', 'Title', 'Author', 'Abstract', 'Section', 'Unknown', 'Metadata' # Add more as needed
        }
        self._docling_document_cache = {} # Cache the output doc per image/options if needed

    def is_available(self) -> bool:
        """Check if docling is installed."""
        return DocumentConverter is not None

    def _get_cache_key(self, options: BaseLayoutOptions) -> str:
        """Generate cache key based on device and potentially converter args."""
        if not isinstance(options, DoclingLayoutOptions):
             options = DoclingLayoutOptions(device=options.device, extra_args=options.extra_args)

        device_key = str(options.device).lower() if options.device else 'default_device'
        # Include hash of extra_args if they affect model loading/converter init
        extra_args_key = hash(frozenset(options.extra_args.items()))
        return f"{self.__class__.__name__}_{device_key}_{extra_args_key}"

    def _load_model_from_options(self, options: BaseLayoutOptions) -> Any:
        """Load the Docling DocumentConverter."""
        if not self.is_available():
            raise RuntimeError("Docling dependency not installed.")

        if not isinstance(options, DoclingLayoutOptions):
            raise TypeError("Incorrect options type provided for Docling model loading.")

        self.logger.info("Initializing Docling DocumentConverter...")
        try:
            # Pass device if converter accepts it, otherwise handle via extra_args
            converter_args = options.extra_args.copy()

            converter = DocumentConverter(**converter_args)
            self.logger.info("Docling DocumentConverter initialized.")
            return converter
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling DocumentConverter: {e}", exc_info=True)
            raise

    def detect(self, image: Image.Image, options: BaseLayoutOptions) -> List[Dict[str, Any]]:
        """Detect document structure and text using Docling."""
        if not self.is_available():
            raise RuntimeError("Docling dependency not installed.")

        if not isinstance(options, DoclingLayoutOptions):
             self.logger.warning("Received BaseLayoutOptions, expected DoclingLayoutOptions. Using defaults.")
             options = DoclingLayoutOptions(
                 confidence=options.confidence, classes=options.classes,
                 exclude_classes=options.exclude_classes, device=options.device,
                 extra_args=options.extra_args, verbose=options.extra_args.get('verbose', False)
             )

        # Validate classes before proceeding (note: Docling classes are case-sensitive)
        # self.validate_classes(options.classes or []) # Validation might be tricky due to case sensitivity
        # if options.exclude_classes:
        #     self.validate_classes(options.exclude_classes)

        # Get the cached/loaded converter instance
        converter = self._get_model(options)

        # Docling convert method requires an image path. Save temp file.
        detections = []
        docling_doc = None # To store the result
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = os.path.join(temp_dir, f"docling_input_{os.getpid()}.png")
            try:
                self.logger.debug(f"Saving temporary image for Docling detector to: {temp_image_path}")
                image.convert("RGB").save(temp_image_path) # Ensure RGB

                # Convert the document using Docling's DocumentConverter
                self.logger.debug("Running Docling conversion...")
                # Docling convert returns a Result object with a 'document' attribute
                result = converter.convert(temp_image_path)
                docling_doc = result.document # Store the DoclingDocument
                self.logger.info(f"Docling conversion complete.")

                # Convert Docling document to our detection format
                detections = self._convert_docling_to_detections(docling_doc, options)

            except Exception as e:
                self.logger.error(f"Error during Docling detection: {e}", exc_info=True)
                raise # Re-raise the exception
            finally:
                 # Ensure temp file is removed
                 if os.path.exists(temp_image_path):
                      try: os.remove(temp_image_path)
                      except OSError as e_rm: self.logger.warning(f"Could not remove temp file {temp_image_path}: {e_rm}")

        # Cache the docling document if needed elsewhere (maybe associate with page?)
        # self._docling_document_cache[image_hash] = docling_doc # Needs a way to key this

        self.logger.info(f"Docling detected {len(detections)} layout elements matching criteria.")
        return detections

    def _convert_docling_to_detections(self, doc, options: DoclingLayoutOptions) -> List[Dict[str, Any]]:
        """Convert a Docling document to our standard detection format."""
        if not doc or not hasattr(doc, 'pages') or not doc.pages:
            self.logger.warning("Invalid or empty Docling document for conversion.")
            return []

        detections = []
        id_to_detection_index = {} # Map Docling ID to index in detections list

        # Prepare normalized class filters once
        normalized_classes_req = {self._normalize_class_name(c) for c in options.classes} if options.classes else None
        normalized_classes_excl = {self._normalize_class_name(c) for c in options.exclude_classes} if options.exclude_classes else set()

        # --- Iterate through elements using Docling's structure ---
        # This requires traversing the hierarchy (e.g., doc.body.children)
        # or iterating through specific lists like doc.texts, doc.tables etc.
        elements_to_process = []
        if hasattr(doc, 'texts'): elements_to_process.extend(doc.texts)
        if hasattr(doc, 'tables'): elements_to_process.extend(doc.tables)
        if hasattr(doc, 'pictures'): elements_to_process.extend(doc.pictures)
        # Add other element types from DoclingDocument as needed

        self.logger.debug(f"Converting {len(elements_to_process)} Docling elements...")

        for elem in elements_to_process:
            try:
                # Get Provenance (bbox and page number)
                if not hasattr(elem, 'prov') or not elem.prov: continue
                prov = elem.prov[0] # Use first provenance
                if not hasattr(prov, 'bbox') or not prov.bbox: continue
                bbox = prov.bbox
                page_no = prov.page_no

                # Get Page Dimensions (crucial for coordinate conversion)
                if not hasattr(doc.pages.get(page_no), 'size'): continue
                page_height = doc.pages[page_no].size.height
                page_width = doc.pages[page_no].size.width # Needed? Bbox seems absolute

                # Convert coordinates from Docling's system (often bottom-left origin)
                # to standard top-left origin (0,0 at top-left)
                # Docling Bbox: l, b, r, t (relative to bottom-left)
                x0 = float(bbox.l)
                x1 = float(bbox.r)
                # Convert y: top_y = page_height - bottom_left_t
                #            bottom_y = page_height - bottom_left_b
                y0 = float(page_height - bbox.t) # Top y
                y1 = float(page_height - bbox.b) # Bottom y

                # Ensure y0 < y1
                if y0 > y1: y0, y1 = y1, y0
                # Ensure x0 < x1
                if x0 > x1: x0, x1 = x1, x0

                # Get Class Label
                label_orig = str(getattr(elem, 'label', 'Unknown')) # Default if no label
                normalized_label = self._normalize_class_name(label_orig)

                # Apply Class Filtering
                if normalized_classes_req and normalized_label not in normalized_classes_req: continue
                if normalized_label in normalized_classes_excl: continue

                # Get Confidence (Docling often doesn't provide per-element confidence)
                confidence = getattr(elem, 'confidence', 0.95) # Assign default confidence
                if confidence < options.confidence: continue # Apply confidence threshold

                # Get Text Content
                text_content = getattr(elem, 'text', None)

                # Get IDs for hierarchy
                docling_id = getattr(elem, 'self_ref', None)
                parent_id_obj = getattr(elem, 'parent', None)
                parent_id = getattr(parent_id_obj, 'self_ref', None) if parent_id_obj else None

                # Create Detection Dictionary
                detection = {
                    'bbox': (x0, y0, x1, y1),
                    'class': label_orig,
                    'normalized_class': normalized_label,
                    'confidence': confidence,
                    'text': text_content,
                    'docling_id': docling_id,
                    'parent_id': parent_id,
                    'page_number': page_no, # Add page number if useful
                    'source': 'layout',
                    'model': 'docling'
                }
                detections.append(detection)

                # Store index for hierarchy linking (if needed later)
                # if docling_id: id_to_detection_index[docling_id] = len(detections) - 1

            except Exception as conv_e:
                 self.logger.warning(f"Could not convert Docling element: {elem}. Error: {conv_e}")
                 continue

        return detections

    def get_docling_document(self, image: Image.Image, options: BaseLayoutOptions):
        """
        Get the raw DoclingDocument object after running detection.
        Ensures detection is run if not already cached for these options/image.
        """
        # This requires caching the doc based on image/options or re-running.
        # For simplicity, let's just re-run detect if needed.
        self.logger.warning("get_docling_document: Re-running detection to ensure document is generated.")
        self.detect(image, options) # Run detect to populate internal doc
        return getattr(self, '_docling_document', None) # Return the stored doc

