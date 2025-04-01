import logging
from typing import List, Dict, Any
from ..elements.region import Region

def convert_to_regions(page: Any, detections: List[Dict[str, Any]], 
                      scale_factor: float = 1.0) -> List[Region]:
    """
    Convert layout detections to Region objects.
    
    Args:
        page: Page object to create regions for
        detections: List of detection dictionaries
        scale_factor: Factor to scale coordinates from image to PDF space
        
    Returns:
        List of Region objects with layout metadata
    """
    conversion_logger = logging.getLogger("natural_pdf.analyzers.layout.convert")
    conversion_logger.debug(f"Converting {len(detections)} detections to regions with scale {scale_factor}")
    regions = []
    
    for det in detections:
        # Extract detection info
        x_min, y_min, x_max, y_max = det['bbox']
        
        # Ensure coordinates are in proper order (min values are smaller)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Scale coordinates from image to PDF space
        if scale_factor != 1.0:
            x_min *= scale_factor
            y_min *= scale_factor
            x_max *= scale_factor
            y_max *= scale_factor
            
        # Create region with metadata
        region = Region(page, (x_min, y_min, x_max, y_max))
        region.region_type = det['class']
        region.confidence = det['confidence']
        region.normalized_type = det['normalized_class']
        
        # Add source info - important for filtering
        region.source = det.get('source', 'detected')
        region.model = det.get('model', 'unknown')
        
        # Add additional metadata if available
        for key, value in det.items():
            if key not in ('bbox', 'class', 'confidence', 'normalized_class', 'source', 'model'):
                setattr(region, key, value)
        
        regions.append(region)
        
    conversion_logger.debug(f"Created {len(regions)} region objects from {len(detections)} detections")
    return regions