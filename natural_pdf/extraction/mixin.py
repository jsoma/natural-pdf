import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type

from pydantic import BaseModel

# Avoid circular import
if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.extraction.result import StructuredDataResult

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURED_KEY = "structured"  # Define default key


class ExtractionMixin(ABC):
    """
    Mixin class providing structured data extraction capabilities to elements.
    Assumes the inheriting class has `extract_text(**kwargs)` and `to_image(**kwargs)` methods.
    """

    def _get_extraction_content(self, using: str = "text", **kwargs) -> Any:
        """
        Retrieves the content (text or image) for extraction.

        Args:
            using: 'text' or 'vision'
            **kwargs: Additional arguments passed to extract_text or to_image

        Returns:
            str: Extracted text if using='text'
            PIL.Image.Image: Rendered image if using='vision'
            None: If content cannot be retrieved
        """
        if not hasattr(self, "extract_text") or not callable(self.extract_text):
            logger.error(f"ExtractionMixin requires 'extract_text' method on {self!r}")
            return None
        if not hasattr(self, "to_image") or not callable(self.to_image):
            logger.error(f"ExtractionMixin requires 'to_image' method on {self!r}")
            return None

        try:
            if using == "text":
                layout = kwargs.pop("layout", True)
                return self.extract_text(layout=layout, **kwargs)
            elif using == "vision":
                resolution = kwargs.pop("resolution", 72)
                include_highlights = kwargs.pop("include_highlights", False)
                labels = kwargs.pop("labels", False)
                return self.to_image(
                    resolution=resolution,
                    include_highlights=include_highlights,
                    labels=labels,
                    **kwargs,
                )
            else:
                logger.error(f"Unsupported value for 'using': {using}")
                return None
        except Exception as e:
            logger.error(f"Error getting {using} content from {self!r}: {e}")
            return None

    def extract(
        self: Any,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str = DEFAULT_STRUCTURED_KEY,  # Default key
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        overwrite: bool = False,  # Add overwrite parameter
        **kwargs,
    ) -> Any:
        """
        Extracts structured data according to the provided schema.

        Results are stored in the element's `analyses` dictionary.

        Args:
            schema: Pydantic model class defining the desired structure
            client: Initialized LLM client
            analysis_key: Key to store the result under in `analyses`. Defaults to "default-structured".
            prompt: Optional user-provided prompt for the LLM
            using: Modality ('text' or 'vision')
            model: Optional specific LLM model identifier
            overwrite: If True, allow overwriting an existing result at `analysis_key`.
            **kwargs: Additional parameters for extraction

        Returns:
            Self for method chaining
        """
        if not analysis_key:
            raise ValueError("analysis_key cannot be empty for extract operation")

        # --- Overwrite Check --- #
        if not hasattr(self, "analyses") or self.analyses is None:
            self.analyses = {}

        if analysis_key in self.analyses and not overwrite:
            raise ValueError(
                f"Analysis key '{analysis_key}' already exists in analyses. "
                f"Use overwrite=True to replace it. Available keys: {list(self.analyses.keys())}"
            )
        # --- End Overwrite Check --- #

        # Determine PDF instance to get manager
        pdf_instance = None

        if hasattr(self, "get_manager") and callable(self.get_manager):
            # Handle case where self is the PDF instance itself
            pdf_instance = self
            logger.debug(f"Manager access via self ({type(self).__name__})")
        elif (
            hasattr(self, "pdf")
            and hasattr(self.pdf, "get_manager")
            and callable(self.pdf.get_manager)
        ):
            # Handle Page or other elements with direct .pdf reference
            pdf_instance = self.pdf
            logger.debug(f"Manager access via self.pdf ({type(self).__name__})")
        elif (
            hasattr(self, "page")
            and hasattr(self.page, "pdf")
            and hasattr(self.page.pdf, "get_manager")
            and callable(self.page.pdf.get_manager)
        ):
            # Handle Region or other elements with .page.pdf reference
            pdf_instance = self.page.pdf
            logger.debug(f"Manager access via self.page.pdf ({type(self).__name__})")
        else:
            logger.error(
                f"Could not find get_manager on {type(self).__name__}, self.pdf, or self.page.pdf"
            )
            raise RuntimeError(
                f"Cannot access PDF manager: {type(self).__name__} lacks necessary references"
            )

        try:
            manager = pdf_instance.get_manager("structured_data")
        except Exception as e:
            raise RuntimeError(f"Failed to get StructuredDataManager: {e}")

        if not manager or not manager.is_available():
            raise RuntimeError("StructuredDataManager is not available")

        # Get content
        layout_for_text = kwargs.pop("layout", True)
        content = self._get_extraction_content(
            using=using, layout=layout_for_text, **kwargs
        )  # Pass kwargs

        if content is None or (
            using == "text" and isinstance(content, str) and not content.strip()
        ):
            logger.warning(f"No content available for extraction (using='{using}') on {self!r}")
            # Import here to avoid circularity at module level
            from natural_pdf.extraction.result import StructuredDataResult

            result = StructuredDataResult(
                data=None,
                success=False,
                error_message=f"No content available for extraction (using='{using}')",
                model=model,  # Use model requested, even if failed
            )
        else:
            result = manager.extract(
                content=content,
                schema=schema,
                client=client,
                prompt=prompt,
                using=using,
                model=model,
                **kwargs,
            )

        # Store the result
        self.analyses[analysis_key] = result
        logger.info(
            f"Stored extraction result under key '{analysis_key}' (Success: {result.success})"
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
            raise ValueError(
                f"Stored result for '{target_key}' indicates a failed extraction attempt. "
                f"Error: {result.error_message}"
            )

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
