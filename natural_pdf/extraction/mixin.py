import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from pydantic import BaseModel

from natural_pdf.qa.qa_result import QAResult
from natural_pdf.services.base import resolve_service

# Avoid circular import
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURED_KEY = "structured"  # Define default key


class ExtractionMixin(ABC):
    """Mixin class providing structured data extraction capabilities to elements.

    This mixin adds AI-powered structured data extraction functionality to pages,
    regions, and elements, enabling extraction of specific data fields using
    Pydantic schemas and large language models. It supports both text-based and
    vision-based extraction modes.

    The mixin integrates with the StructuredDataManager to handle LLM interactions
    and provides schema validation using Pydantic models. Extracted data is
    automatically validated against the provided schema and stored with
    confidence metrics and metadata.

    Extraction modes:
    - Text-based: Uses extracted text content for LLM processing
    - Vision-based: Uses rendered images for multimodal LLM analysis
    - Automatic: Selects best mode based on content and model capabilities

    Host class requirements:
    - Must implement extract_text(**kwargs) -> str
    - Must implement render(**kwargs) -> PIL.Image
    - Must have access to StructuredDataManager (usually via parent PDF)

    Example:
        ```python
        from pydantic import BaseModel

        class InvoiceData(BaseModel):
            invoice_number: str
            total_amount: float
            due_date: str
            vendor_name: str

        pdf = npdf.PDF("invoice.pdf")
        page = pdf.pages[0]

        # Extract structured data
        invoice = page.extract_structured_data(InvoiceData)
        print(f"Invoice {invoice.data.invoice_number}: ${invoice.data.total_amount}")

        # Region-specific extraction
        header_region = page.find('text:contains("Invoice")').above()
        header_data = header_region.extract_structured_data(InvoiceData)
        ```

    Note:
        Structured extraction requires a compatible LLM to be configured in the
        StructuredDataManager. Results include confidence scores and validation
        metadata for quality assessment.
    """

    def _get_extraction_content(self: Any, using: str = "text", **kwargs) -> Any:
        """
        Retrieves the content (text or image) for extraction.

        Args:
            using: 'text' or 'vision'
            **kwargs: Additional arguments passed to extract_text or render

        Returns:
            str: Extracted text if using='text'
            PIL.Image.Image: Rendered image if using='vision'
            None: If content cannot be retrieved
        """
        try:
            if using == "text":
                if not hasattr(self, "extract_text") or not callable(self.extract_text):
                    logger.error(f"ExtractionMixin requires 'extract_text' method on {self!r}")
                    return None
                layout = kwargs.pop("layout", True)
                return self.extract_text(layout=layout, **kwargs)
            elif using == "vision":
                if not hasattr(self, "render") or not callable(self.render):
                    logger.error(f"ExtractionMixin requires 'render' method on {self!r}")
                    return None
                resolution = kwargs.pop("resolution", 72)
                include_highlights = kwargs.pop("include_highlights", False)
                labels = kwargs.pop("labels", False)
                return self.render(
                    resolution=resolution,
                    **kwargs,
                )
            else:
                logger.error(f"Unsupported value for 'using': {using}")
                return None
        except Exception as e:
            import warnings

            warnings.warn(
                f"Error getting {using} content from {self!r}: {e}",
                RuntimeWarning,
            )
            raise

    def extract(
        self: Any,
        schema: Type[BaseModel],
        client: Any = None,
        analysis_key: str = DEFAULT_STRUCTURED_KEY,
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        engine: Optional[str] = None,
        overwrite: bool = True,
        **kwargs,
    ) -> Any:
        """Extract structured data via the extraction service."""
        resolve_service(self, "extraction").extract(
            host=self,
            schema=schema,
            client=client,
            analysis_key=analysis_key,
            prompt=prompt,
            using=using,
            model=model,
            engine=engine,
            overwrite=overwrite,
            **kwargs,
        )
        return self

    def extracted(
        self, field_name: Optional[str] = None, analysis_key: Optional[str] = None
    ) -> Any:
        """
        Convenience method to access results from structured data extraction.

        Args:
            field_name: The specific field to retrieve from the extracted data dictionary.
                        If None, returns the entire data dictionary.
            analysis_key: The key under which the extraction result was stored in `analyses`.
                          If None, defaults to "default-structured".

        Returns:
            The requested field value, the entire data dictionary, or raises an error.

        Raises:
            KeyError: If the specified `analysis_key` is not found in `analyses`.
            ValueError: If the stored result for `analysis_key` indicates a failed extraction.
            AttributeError: If the element does not have an `analyses` attribute.
            KeyError: (Standard Python) If `field_name` is specified but not found in the data.
        """
        target_key = analysis_key if analysis_key is not None else DEFAULT_STRUCTURED_KEY

        if not hasattr(self, "analyses") or self.analyses is None:
            raise AttributeError(f"{type(self).__name__} object has no 'analyses' attribute yet.")

        if target_key not in self.analyses:
            available_keys = list(self.analyses.keys())
            raise KeyError(
                f"Extraction '{target_key}' not found in analyses. "
                f"Available extractions: {available_keys}"
            )

        # Import here to avoid circularity and allow type checking
        from natural_pdf.extraction.result import StructuredDataResult

        result: StructuredDataResult = self.analyses[target_key]

        if not isinstance(result, StructuredDataResult):
            logger.warning(
                f"Item found at key '{target_key}' is not a StructuredDataResult (type: {type(result)}). Cannot process."
            )
            raise TypeError(
                f"Expected a StructuredDataResult at key '{target_key}', found {type(result).__name__}"
            )

        if not result.success:
            # Return None for failed extractions to allow batch processing to continue
            logger.warning(
                f"Extraction '{target_key}' failed: {result.error_message}. Returning None."
            )
            return None

        if result.data is None:
            # This case might occur if success=True but data is somehow None
            raise ValueError(
                f"Extraction result for '{target_key}' has no data available, despite success flag."
            )

        if field_name is None:
            # Return the whole data object (Pydantic model instance or dict)
            return result.data
        else:
            # Try dictionary key access first, then attribute access
            if isinstance(result.data, dict):
                try:
                    return result.data[field_name]
                except KeyError:
                    available_keys = list(result.data.keys())
                    raise KeyError(
                        f"Field/Key '{field_name}' not found in extracted dictionary "
                        f"for key '{target_key}'. Available keys: {available_keys}"
                    )
            else:
                # Assume it's an object, try attribute access
                try:
                    return getattr(result.data, field_name)
                except AttributeError:
                    # Try to get available fields from the object
                    available_fields = []
                    if hasattr(result.data, "model_fields"):  # Pydantic v2
                        available_fields = list(result.data.model_fields.keys())
                    elif hasattr(result.data, "__fields__"):  # Pydantic v1
                        available_fields = list(result.data.__fields__.keys())
                    elif hasattr(result.data, "__dict__"):  # Fallback
                        available_fields = list(result.data.__dict__.keys())

                    raise AttributeError(
                        f"Field/Attribute '{field_name}' not found on extracted object of type {type(result.data).__name__} "
                        f"for key '{target_key}'. Available fields/attributes: {available_fields}"
                    )
                except Exception as e:  # Catch other potential errors during getattr
                    raise TypeError(
                        f"Could not access field/attribute '{field_name}' on extracted data for key '{target_key}' (type: {type(result.data).__name__}). Error: {e}"
                    ) from e

    # ------------------------------------------------------------------
    # Internal helper methods now live inside ExtractionService
    analyses: Optional[Dict[str, Any]] = None
