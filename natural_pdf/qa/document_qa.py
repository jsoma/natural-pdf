import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile
import json

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
            
            # Determine device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Initializing DocumentQA with model {model_name} on {device}")
            
            # Initialize the pipeline
            self.pipe = pipeline("document-question-answering", model=model_name, device=device)
            
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
            if hasattr(element, 'text') and element.text.strip():
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
                
                word_boxes.append([
                    element.text,
                    [x0, top, x1, bottom]
                ])
        
        return word_boxes
    
    def ask(self, 
            image: Union[str, Image.Image, np.ndarray], 
            question: str, 
            word_boxes: List = None,
            min_confidence: float = 0.1,
            debug: bool = False,
            debug_output_dir: str = "output") -> Dict[str, Any]:
        """
        Ask a question about document content.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            question: Question to ask about the document
            word_boxes: Optional pre-extracted word boxes [[text, [x0, y0, x1, y1]], ...]
            min_confidence: Minimum confidence threshold for answers
            debug: Whether to save debug information
            debug_output_dir: Directory to save debug files
            
        Returns:
            Dictionary with answer details: {
                "answer": extracted text,
                "confidence": confidence score,
                "start": start word index,
                "end": end word index
            }
        """
        if not self._is_initialized:
            raise RuntimeError("DocumentQA is not properly initialized")
        
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
        
        # Prepare the query
        query = {
            "image": image_obj,
            "question": question
        }
        
        # Add word boxes if provided
        if word_boxes:
            query["word_boxes"] = word_boxes
            
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
                with open(word_boxes_path, 'w') as f:
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
        
        # Run the query through the pipeline
        logger.info(f"Running document QA pipeline with question: {question}")
        result = self.pipe(query)[0]
        logger.info(f"Raw result: {result}")
        
        # Save the result if debugging
        if debug:
            result_path = os.path.join(debug_output_dir, "debug_qa_result.json")
            with open(result_path, 'w') as f:
                # Convert any non-serializable data
                serializable_result = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v 
                                    for k, v in result.items()}
                json.dump(serializable_result, f, indent=2)
        
        # Check confidence against threshold
        if result["score"] < min_confidence:
            logger.info(f"Answer confidence {result['score']:.4f} below threshold {min_confidence}")
            return {
                "answer": "",
                "confidence": result["score"],
                "start": result.get("start", -1),
                "end": result.get("end", -1),
                "found": False
            }
        
        return {
            "answer": result["answer"],
            "confidence": result["score"],
            "start": result.get("start", 0),
            "end": result.get("end", 0),
            "found": True
        }
        
    
    def ask_pdf_page(self, page, question: str, min_confidence: float = 0.1, debug: bool = False) -> Dict[str, Any]:
        """
        Ask a question about a specific PDF page.
        
        Args:
            page: natural_pdf.core.page.Page object
            question: Question to ask about the page
            min_confidence: Minimum confidence threshold for answers
            
        Returns:
            Dictionary with answer details
        """
        # Ensure we have text elements on the page
        if not page.find_all('text'):
            # Apply OCR if no text is available
            logger.info(f"No text elements found on page {page.index}, applying OCR")
            page.apply_ocr()
        
        # Extract word boxes
        elements = page.find_all('text')
        word_boxes = self._get_word_boxes_from_elements(elements, offset_x=0, offset_y=0)
        
        # Generate a high-resolution image of the page
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save a high resolution image (300 DPI)
        page_image = page.to_image(resolution=300, include_highlights=False)
        page_image.save(temp_path)
        
        try:
            # Ask the question
            result = self.ask(
                image=temp_path,
                question=question,
                word_boxes=word_boxes,
                min_confidence=min_confidence,
                debug=debug
            )
            
            # Add page reference to the result
            result["page_num"] = page.index
            
            # Add element references if possible
            if result.get("found", False) and "start" in result and "end" in result:
                start_idx = result["start"]
                end_idx = result["end"]
                
                # Make sure we have valid indices and elements to work with
                if elements and 0 <= start_idx < len(word_boxes) and 0 <= end_idx < len(word_boxes):
                    # Find the actual source elements in the original list
                    # Since word_boxes may have filtered out some elements, we need to map indices
                    
                    # Get the text from result word boxes
                    matched_texts = [wb[0] for wb in word_boxes[start_idx:end_idx+1]]
                    
                    # Find corresponding elements in the full element list
                    source_elements = []
                    for element in elements:
                        if hasattr(element, 'text') and element.text in matched_texts:
                            source_elements.append(element)
                            # Remove from matched texts to avoid duplicates
                            if element.text in matched_texts:
                                matched_texts.remove(element.text)
                    
                    result["source_elements"] = source_elements
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def ask_pdf_region(self, region, question: str, min_confidence: float = 0.1, debug: bool = False) -> Dict[str, Any]:
        """
        Ask a question about a specific region of a PDF page.
        
        Args:
            region: natural_pdf.elements.region.Region object
            question: Question to ask about the region
            min_confidence: Minimum confidence threshold for answers
            
        Returns:
            Dictionary with answer details
        """
        # Get all text elements within the region
        elements = region.find_all('text')
        
        # Apply OCR if needed
        if not elements:
            logger.info(f"No text elements found in region, applying OCR")
            elements = region.apply_ocr()
        
        # Extract word boxes adjusted for the cropped region
        x0, top = int(region.x0), int(region.top)
        word_boxes = self._get_word_boxes_from_elements(elements, offset_x=x0, offset_y=top)
        
        # Generate a cropped image of the region
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Get page image at high resolution - this returns a PIL Image directly
        page_image = region.page.to_image(resolution=300, include_highlights=False)
        
        # Crop to region
        x0, top, x1, bottom = int(region.x0), int(region.top), int(region.x1), int(region.bottom)
        region_image = page_image.crop((x0, top, x1, bottom))
        region_image.save(temp_path)
        
        try:
            # Ask the question
            result = self.ask(
                image=temp_path,
                question=question,
                word_boxes=word_boxes,
                min_confidence=min_confidence,
                debug=debug
            )
            
            # Add region reference to the result
            result["region"] = region
            result["page_num"] = region.page.index
            
            # Add element references if possible
            if result.get("found", False) and "start" in result and "end" in result:
                start_idx = result["start"]
                end_idx = result["end"]
                
                # Make sure we have valid indices and elements to work with
                if elements and 0 <= start_idx < len(word_boxes) and 0 <= end_idx < len(word_boxes):
                    # Find the actual source elements in the original list
                    # Since word_boxes may have filtered out some elements, we need to map indices
                    
                    # Get the text from result word boxes
                    matched_texts = [wb[0] for wb in word_boxes[start_idx:end_idx+1]]
                    
                    # Find corresponding elements in the full element list
                    source_elements = []
                    for element in elements:
                        if hasattr(element, 'text') and element.text in matched_texts:
                            source_elements.append(element)
                            # Remove from matched texts to avoid duplicates
                            if element.text in matched_texts:
                                matched_texts.remove(element.text)
                    
                    result["source_elements"] = source_elements
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)