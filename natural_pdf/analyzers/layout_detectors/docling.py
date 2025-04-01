import logging
from typing import List, Dict, Any, Optional

from .base import LayoutDetector

class DoclingLayoutDetector(LayoutDetector):
    """
    Document layout and text recognition using Docling.
    
    Docling provides a hierarchical document understanding system that can analyze:
    - Document structure (headers, text, figures, tables)
    - Text content via integrated OCR
    - Hierarchical relationships between document elements
    """
    
    def __init__(self, verbose=False, **kwargs):
        """
        Initialize the Docling document analyzer.
        
        Args:
            verbose: Whether to enable verbose logging
            **kwargs: Additional parameters to pass to DocumentConverter
        """
        # Set up logger with optional verbose mode
        import logging
        self.logger = logging.getLogger("natural_pdf.analyzers.layout.docling")
        self.original_level = self.logger.level
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        super().__init__()
        self.verbose = verbose
        self.converter_kwargs = kwargs
        self._docling_document = None
        self._converter = None
        
    def __del__(self):
        # Restore the original logging level when done
        if hasattr(self, 'logger') and hasattr(self, 'original_level'):
            self.logger.setLevel(self.original_level)
            
    @property
    def converter(self):
        """Lazy-load the DocumentConverter on first use."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self.logger.debug("Initializing Docling DocumentConverter")
                self._converter = DocumentConverter(**self.converter_kwargs)
            except ImportError:
                raise ImportError(
                    "Docling integration requires docling. "
                    "Install with: pip install docling"
                )
        return self._converter
    
    def detect(self, image_path, confidence=0.5, classes=None, exclude_classes=None):
        """
        Detect document structure and text using Docling.
        
        Args:
            image_path: Path to the image or PDF to analyze
            confidence: Minimum confidence threshold for detections (not used by Docling)
            classes: Specific classes to detect (used for filtering)
            exclude_classes: Classes to exclude from detection (used for filtering)
            
        Returns:
            List of detection dictionaries with hierarchical information
        """
        self.logger.info(f"Processing {image_path} with Docling")
        
        try:
            # Convert the document using Docling's DocumentConverter
            result = self.converter.convert(image_path)
            doc = result.document
            
            # Store for later use
            self._docling_document = doc
            self.logger.info(f"Docling document created with {len(doc.body.children)} top-level elements")
            
            # Convert Docling document to our detection format
            detections = self._convert_docling_to_detections(doc, confidence, classes, exclude_classes)
            
            return detections
        except Exception as e:
            self.logger.error(f"Error processing with Docling: {e}")
            raise
    
    def _convert_docling_to_detections(self, doc, confidence, classes, exclude_classes):
        """
        Convert a Docling document to our standard detection format.
        
        Args:
            doc: DoclingDocument object
            confidence: Confidence threshold to apply (not used by Docling)
            classes: Classes to include (if specified)
            exclude_classes: Classes to exclude
            
        Returns:
            List of detection dictionaries with hierarchy information
        """
        if not doc or not hasattr(doc, 'body') or not hasattr(doc.body, 'children'):
            self.logger.warning("Invalid or empty Docling document")
            return []
            
        detections = []
        id_to_detection = {}  # Map from Docling ID to detection index
        
        # Process text elements
        if hasattr(doc, 'texts') and doc.texts:
            self.logger.debug(f"Processing {len(doc.texts)} text elements")
            
            # First pass: create detections for all text elements
            for text_elem in doc.texts:
                # Skip if no provenance information
                if not hasattr(text_elem, 'prov') or not text_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = text_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792  # Default letter size
                # Already in top-left coordinates
                t = page_height - bbox.t
                b = page_height - bbox.b
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                # Get the label and normalize it
                label = str(text_elem.label) if hasattr(text_elem, 'label') else 'text'
                normalized_label = self._normalize_class_name(label)
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence for Docling
                    'text': text_elem.text if hasattr(text_elem, 'text') else None,
                    'docling_id': text_elem.self_ref if hasattr(text_elem, 'self_ref') else None,
                    'parent_id': text_elem.parent.self_ref if hasattr(text_elem, 'parent') and hasattr(text_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
        
        # Process pictures if available
        if hasattr(doc, 'pictures') and doc.pictures:
            self.logger.debug(f"Processing {len(doc.pictures)} picture elements")
            
            for pic_elem in doc.pictures:
                # Skip if no provenance information
                if not hasattr(pic_elem, 'prov') or not pic_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = pic_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792
                # In BOTTOMLEFT system, bbox.t is distance from bottom (higher value = higher on page)
                # In TOPLEFT system, we need distance from top (convert using page_height)
                t = page_height - bbox.t  # Correct: Top is page_height minus the top in BOTTOMLEFT
                b = page_height - bbox.b  # Correct: Bottom is page_height minus the bottom in BOTTOMLEFT
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                label = 'figure'  # Default label for pictures
                normalized_label = 'figure'
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence
                    'docling_id': pic_elem.self_ref if hasattr(pic_elem, 'self_ref') else None,
                    'parent_id': pic_elem.parent.self_ref if hasattr(pic_elem, 'parent') and hasattr(pic_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
                    
        # Process tables if available
        if hasattr(doc, 'tables') and doc.tables:
            self.logger.debug(f"Processing {len(doc.tables)} table elements")
            
            for table_elem in doc.tables:
                # Skip if no provenance information
                if not hasattr(table_elem, 'prov') or not table_elem.prov:
                    continue
                    
                # Get the bounding box
                prov = table_elem.prov[0]  # Take first provenance entry
                if not hasattr(prov, 'bbox') or not prov.bbox:
                    continue
                
                bbox = prov.bbox
                
                # Convert from bottom-left to top-left coordinates
                page_height = doc.pages.get(prov.page_no).size.height if hasattr(doc, 'pages') else 792
                # In BOTTOMLEFT system, bbox.t is distance from bottom (higher value = higher on page)
                # In TOPLEFT system, we need distance from top (convert using page_height)
                t = page_height - bbox.t  # Correct: Top is page_height minus the top in BOTTOMLEFT
                b = page_height - bbox.b  # Correct: Bottom is page_height minus the bottom in BOTTOMLEFT
                
                # Ensure top is always less than bottom for PIL coordinates
                if t > b:
                    t, b = b, t
                
                label = 'table'  # Default label for tables
                normalized_label = 'table'
                
                # Skip if filtered by class
                if classes and normalized_label not in classes:
                    continue
                if exclude_classes and normalized_label in exclude_classes:
                    continue
                
                # Create detection
                detection = {
                    'bbox': (bbox.l, t, bbox.r, b),
                    'class': label,
                    'normalized_class': normalized_label,
                    'confidence': 0.95,  # Default confidence
                    'docling_id': table_elem.self_ref if hasattr(table_elem, 'self_ref') else None,
                    'parent_id': table_elem.parent.self_ref if hasattr(table_elem, 'parent') and hasattr(table_elem.parent, 'self_ref') else None,
                    'model': 'docling'
                }
                
                detections.append(detection)
                
                # Track by ID for hierarchy reconstruction
                if detection['docling_id']:
                    id_to_detection[detection['docling_id']] = len(detections) - 1
        
        self.logger.info(f"Created {len(detections)} detections from Docling document")
        return detections
    
    def get_docling_document(self):
        """Get the original Docling document for advanced usage."""
        return self._docling_document
