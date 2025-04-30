import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ExportMixin:
    """
    Mixin for exporting analyses from collections of elements.

    This mixin is designed to be used with PDF, PDFCollection,
    PageCollection, and ElementCollection classes.
    """

    def export_analyses(
        self,
        output_path: str,
        analysis_keys: Union[str, List[str]],
        format: str = "json",
        include_content: bool = True,
        include_images: bool = False,
        image_dir: Optional[str] = None,
        image_format: str = "jpg",
        image_resolution: int = 72,
        overwrite: bool = True,
        **kwargs,
    ) -> str:
        """
        Export analysis results to a file.

        Args:
            output_path: Path to save the export file
            analysis_keys: Key(s) in the analyses dictionary to export
            format: Export format ('json', 'csv', 'excel')
            include_content: Whether to include extracted text
            include_images: Whether to export images of elements
            image_dir: Directory to save images (created if doesn't exist)
            image_format: Format to save images ('jpg', 'png')
            image_resolution: Resolution for exported images
            overwrite: Whether to overwrite existing files
            **kwargs: Additional format-specific options

        Returns:
            Path to the exported file
        """
        # Convert single key to list for consistency
        if isinstance(analysis_keys, str):
            analysis_keys = [analysis_keys]

        # Create output directory
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        # Check if file exists and handle overwrite
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file {output_path} already exists and overwrite=False")

        # Prepare image directory if needed
        if include_images:
            if image_dir is None:
                image_dir = output_path.parent / f"{output_path.stem}_images"
            os.makedirs(image_dir, exist_ok=True)
            image_dir = Path(image_dir)  # Convert to Path object

        # Gather data from collection
        data = self._gather_analysis_data(
            analysis_keys=analysis_keys,
            include_content=include_content,
            include_images=include_images,
            image_dir=image_dir,
            image_format=image_format,
            image_resolution=image_resolution,
        )

        # Export based on format
        if format.lower() == "json":
            return self._export_to_json(data, output_path, **kwargs)
        elif format.lower() == "csv":
            return self._export_to_csv(data, output_path, **kwargs)
        elif format.lower() == "excel":
            return self._export_to_excel(data, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _gather_analysis_data(
        self,
        analysis_keys: List[str],
        include_content: bool,
        include_images: bool,
        image_dir: Optional[Path],
        image_format: str,
        image_resolution: int,
    ) -> List[Dict[str, Any]]:
        """
        Gather analysis data from elements in the collection.

        This method should be implemented by each collection class.
        """
        raise NotImplementedError("Subclasses must implement _gather_analysis_data")

    def _export_to_json(self, data: List[Dict[str, Any]], output_path: Path, **kwargs) -> str:
        """Export data to JSON format."""
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, **kwargs)
        logger.info(f"Exported analysis data to {output_path}")
        return str(output_path)

    def _export_to_csv(self, data: List[Dict[str, Any]], output_path: Path, **kwargs) -> str:
        """Export data to CSV format."""
        try:
            import pandas as pd

            # Normalize nested data
            df = pd.json_normalize(data)
            df.to_csv(output_path, index=False, **kwargs)
            logger.info(f"Exported analysis data to {output_path}")
            return str(output_path)
        except ImportError:
            raise ImportError("Pandas is required for CSV export. Install with: pip install pandas")

    def _export_to_excel(self, data: List[Dict[str, Any]], output_path: Path, **kwargs) -> str:
        """Export data to Excel format."""
        try:
            import pandas as pd

            # Normalize nested data
            df = pd.json_normalize(data)
            df.to_excel(output_path, index=False, **kwargs)
            logger.info(f"Exported analysis data to {output_path}")
            return str(output_path)
        except ImportError:
            raise ImportError(
                "Pandas and openpyxl are required for Excel export. Install with: pip install pandas openpyxl"
            )
