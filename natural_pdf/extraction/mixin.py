import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type

from pydantic import BaseModel, Field, create_model

# Avoid circular import
if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.extraction.result import StructuredDataResult

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

    def _get_extraction_content(self, using: str = "text", **kwargs) -> Any:
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
        analysis_key: str = DEFAULT_STRUCTURED_KEY,  # Default key
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        engine: Optional[str] = None,  # NEW: choose between 'llm' and 'doc_qa'
        overwrite: bool = True,  # Overwrite by default
        **kwargs,
    ) -> Any:
        """
        Extracts structured data according to the provided schema.

        Results are stored in the element's `analyses` dictionary.

        Args:
            schema: Either a Pydantic model class defining the desired structure, or an
                    iterable (e.g. list) of field names. When iterable is supplied a
                    temporary Pydantic model of string fields is created automatically.
            client: Initialized LLM client (required for LLM engine only)
            analysis_key: Key to store the result under in `analyses`. Defaults to "default-structured".
            prompt: Optional user-provided prompt for the LLM
            using: Modality ('text' or 'vision')
            model: Optional specific LLM model identifier
            engine: Extraction engine to use ("llm" or "doc_qa"). If None, auto-determined.
            overwrite: Whether to overwrite an existing result stored at `analysis_key`. Defaults to True.
            **kwargs: Additional parameters for extraction

        Returns:
            Self for method chaining
        """
        # ------------------------------------------------------------------
        # If the user supplied a plain list/tuple of field names, dynamically
        # build a simple Pydantic model (all `str` fields) so the rest of the
        # pipeline can work unmodified.
        # ------------------------------------------------------------------
        if not isinstance(schema, type):  # not already a class
            if isinstance(schema, Sequence):
                field_names = list(schema)
                if not field_names:
                    raise ValueError("Schema list cannot be empty")

                import re

                field_defs = {}
                for orig_name in field_names:
                    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", orig_name)
                    if safe_name and safe_name[0].isdigit():
                        safe_name = f"_{safe_name}"

                    field_defs[safe_name] = (
                        str,
                        Field(
                            None,
                            description=f"{orig_name}",
                            alias=orig_name,  # allow access via original name
                        ),
                    )

                schema = create_model("DynamicExtractSchema", **field_defs)  # type: ignore[arg-type]
            else:
                raise TypeError(
                    "schema must be a Pydantic model class or a sequence of field names"
                )

        # ------------------------------------------------------------------
        # Resolve which engine to use
        # ------------------------------------------------------------------
        if engine not in (None, "llm", "doc_qa"):
            raise ValueError("engine must be either 'llm', 'doc_qa', or None")

        # Auto-select: LLM when client provided, else Document-QA
        if engine is None:
            engine = "llm" if client is not None else "doc_qa"

        logger.info(f"Extraction engine resolved to '{engine}'")

        if not analysis_key:
            raise ValueError("analysis_key cannot be empty for extract operation")

        # --- Overwrite Check --- #
        if not hasattr(self, "analyses") or self.analyses is None:
            self.analyses = {}

        if analysis_key in self.analyses and not overwrite:
            logger.info(
                f"Extraction for key '{analysis_key}' already exists; returning cached result. "
                "Pass overwrite=True to force re-extraction."
            )
            return self
        # --- End Overwrite Check --- #

        # ------------------------------------------------------------------
        # Delegate to engine-specific helpers and return early
        # ------------------------------------------------------------------
        if engine == "doc_qa":
            self._perform_docqa_extraction(
                schema=schema,
                analysis_key=analysis_key,
                model=model,
                overwrite=overwrite,
                **kwargs,
            )
            return self

        if engine == "llm":
            if client is None:
                raise ValueError("LLM engine selected but no 'client' was provided.")

            self._perform_llm_extraction(
                schema=schema,
                client=client,
                analysis_key=analysis_key,
                prompt=prompt,
                using=using,
                model=model,
                overwrite=overwrite,
                **kwargs,
            )
            return self

        # ------------------------------------------------------------------
        # LLM ENGINE  (existing behaviour)
        # ------------------------------------------------------------------
        if engine == "llm" and client is None:
            raise ValueError("LLM engine selected but no 'client' was provided.")

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
        content = self._get_extraction_content(using=using, **kwargs)  # Pass kwargs

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
                model_used=model,  # Use model requested, even if failed
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
    # Internal helper: Document-QA powered extraction
    # ------------------------------------------------------------------
    def _perform_docqa_extraction(
        self,
        *,
        schema: Type[BaseModel],
        analysis_key: str,
        model: Optional[str] = None,
        overwrite: bool = True,
        min_confidence: float = 0.1,
        debug: bool = False,
        question_map: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Run extraction using the local Document-QA engine.

        Mutates ``self.analyses[analysis_key]`` with a StructuredDataResult.
        """
        question_map = question_map or {}

        try:
            import re

            from pydantic import Field as _Field
            from pydantic import create_model

            from natural_pdf.extraction.result import StructuredDataResult
            from natural_pdf.qa.document_qa import get_qa_engine
        except ImportError as exc:
            raise RuntimeError(
                "Document-QA dependencies missing. Install with `pip install natural-pdf[ai]`."
            ) from exc

        qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()

        # Iterate over schema fields
        if hasattr(schema, "__fields__"):
            fields_iter = schema.__fields__.items()  # Pydantic v1
        else:
            fields_iter = schema.model_fields.items()  # Pydantic v2

        answers: dict = {}
        confidences: dict = {}
        errors: list[str] = []

        # Ensure we can call QA on this object type
        from natural_pdf.core.page import Page as _Page
        from natural_pdf.elements.region import Region as _Region

        if not isinstance(self, (_Page, _Region)):
            raise NotImplementedError(
                "Document-QA extraction is only supported on Page or Region objects."
            )

        for field_name, field_obj in fields_iter:
            display_name = getattr(field_obj, "alias", field_name)

            # Compose question text
            if display_name in question_map:
                question = question_map[display_name]
            else:
                description = None
                if hasattr(field_obj, "field_info") and hasattr(
                    field_obj.field_info, "description"
                ):
                    description = field_obj.field_info.description
                elif hasattr(field_obj, "description"):
                    description = field_obj.description

                question = description or f"What is the {display_name.replace('_', ' ')}?"

            try:
                # Ask via appropriate helper
                if isinstance(self, _Page):
                    qa_resp = qa_engine.ask_pdf_page(
                        self,
                        question,
                        min_confidence=min_confidence,
                        debug=debug,
                    )
                else:  # Region
                    qa_resp = qa_engine.ask_pdf_region(
                        self,
                        question,
                        min_confidence=min_confidence,
                        debug=debug,
                    )

                confidence_val = qa_resp.get("confidence") if qa_resp else None
                answer_val = qa_resp.get("answer") if qa_resp else None

                if confidence_val is not None and confidence_val < min_confidence:
                    answer_val = None

                answers[display_name] = answer_val
                confidences[f"{display_name}_confidence"] = confidence_val

            except Exception as e:  # noqa: BLE001
                logger.error("Doc-QA failed for field '%s': %s", field_name, e)
                errors.append(str(e))
                answers[display_name] = None
                confidences[f"{display_name}_confidence"] = None

        combined = {**answers, **confidences}

        # Build extended model that includes confidence fields
        field_defs_ext = {}
        for orig_key, val in combined.items():
            safe_key = re.sub(r"[^0-9a-zA-Z_]", "_", orig_key)
            if safe_key and safe_key[0].isdigit():
                safe_key = f"_{safe_key}"

            if orig_key.endswith("_confidence"):
                field_defs_ext[safe_key] = (
                    Optional[float],
                    _Field(None, description=f"Confidence for {orig_key}", alias=orig_key),
                )
            else:
                field_defs_ext[safe_key] = (
                    Optional[type(val) if val is not None else str],
                    _Field(None, alias=orig_key),
                )

        ExtendedSchema = create_model(f"{schema.__name__}WithConf", **field_defs_ext)

        try:
            structured_instance = ExtendedSchema(**combined)
            success_flag = not errors
            err_msg = None if not errors else "; ".join(errors)
        except Exception as exc:  # noqa: BLE001
            structured_instance = None
            success_flag = False
            err_msg = str(exc)

        result = StructuredDataResult(
            data=structured_instance if structured_instance is not None else combined,
            success=success_flag,
            error_message=err_msg,
            model_used=getattr(qa_engine, "model_name", None),
        )

        self.analyses[analysis_key] = result

    # ------------------------------------------------------------------
    # Internal helper: LLM powered extraction (existing behaviour)
    # ------------------------------------------------------------------
    def _perform_llm_extraction(
        self,
        *,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str,
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        """Run extraction via the StructuredDataManager (LLM)."""

        from natural_pdf.extraction.result import StructuredDataResult

        # Determine PDF instance to obtain StructuredDataManager
        pdf_instance = None

        if hasattr(self, "get_manager") and callable(self.get_manager):
            pdf_instance = self
        elif hasattr(self, "pdf") and hasattr(self.pdf, "get_manager"):
            pdf_instance = self.pdf
        elif (
            hasattr(self, "page")
            and hasattr(self.page, "pdf")
            and hasattr(self.page.pdf, "get_manager")
        ):
            pdf_instance = self.page.pdf
        else:
            raise RuntimeError("Cannot access PDF manager to perform LLM extraction.")

        manager = pdf_instance.get_manager("structured_data")
        if not manager or not manager.is_available():
            raise RuntimeError("StructuredDataManager is not available")

        # Content preparation
        content = self._get_extraction_content(using=using, **kwargs)

        import warnings

        if content is None or (
            using == "text" and isinstance(content, str) and not content.strip()
        ):
            preview = None
            if isinstance(content, str):
                preview = content[:120]
            msg = (
                f"No content available for extraction (using='{using}'). "
                "Ensure the page has a text layer or render() returns an image. "
                "For scanned PDFs run apply_ocr() or switch to using='vision'. "
                f"Content preview: {preview!r}"
            )
            warnings.warn(msg, RuntimeWarning)

            result = StructuredDataResult(
                data=None,
                success=False,
                error_message=msg,
                model_used=model,
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

        self.analyses[analysis_key] = result
