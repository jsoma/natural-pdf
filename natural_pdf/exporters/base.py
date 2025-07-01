import abc
import logging
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from natural_pdf.core.pdf import PDF
    from natural_pdf.core.pdf_collection import PDFCollection

logger = logging.getLogger(__name__)


class FinetuneExporter(abc.ABC):
    """
    Abstract base class for exporting data suitable for fine-tuning models.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the exporter with format-specific options.
        """
        pass

    @abc.abstractmethod
    def export(self, source: Union["PDF", "PDFCollection", List["PDF"]], output_dir: str, **kwargs):
        """
        Exports the data from the source PDF(s) to the specified output directory
        in a format suitable for fine-tuning a specific model type.

        Args:
            source: The PDF object, PDFCollection, or list of PDF objects to process.
            output_dir: The path to the directory where the exported files will be saved.
            **kwargs: Additional export-time arguments.
        """
        pass

    def _resolve_source_pdfs(
        self, source: Union["PDF", "PDFCollection", List["PDF"]]
    ) -> List["PDF"]:
        """
        Helper to consistently resolve the input source to a list of PDF objects.
        """
        from natural_pdf.core.pdf import PDF  # Avoid circular import at module level
        from natural_pdf.core.pdf_collection import PDFCollection  # Avoid circular import

        pdfs_to_process: List["PDF"] = []
        if isinstance(source, PDF):
            pdfs_to_process = [source]
        elif isinstance(source, PDFCollection):
            pdfs_to_process = source.pdfs
        elif isinstance(source, list) and all(isinstance(p, PDF) for p in source):
            pdfs_to_process = source
        else:
            raise TypeError(
                f"Unsupported source type: {type(source)}. Must be PDF, PDFCollection, or List[PDF]."
            )

        if not pdfs_to_process:
            logger.warning("No PDF documents provided in the source.")

        return pdfs_to_process
