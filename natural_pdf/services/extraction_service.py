from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

from pydantic import BaseModel, Field, create_model

from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.extraction.structured_ops import (
    extract_structured_data,
    structured_data_is_available,
)
from natural_pdf.qa.qa_result import QAResult
from natural_pdf.services.registry import register_delegate

DEFAULT_STRUCTURED_KEY = "structured"

logger = logging.getLogger(__name__)


class ExtractionService:
    """Shared structured extraction helpers for Page/Region hosts."""

    def __init__(self, context):
        self._context = context

    @register_delegate("extraction", "extract")
    def extract(
        self,
        host,
        schema: Union[Type[BaseModel], Sequence[str]],
        client: Any = None,
        analysis_key: str = DEFAULT_STRUCTURED_KEY,
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        engine: Optional[str] = None,
        overwrite: bool = True,
        **kwargs,
    ):
        """Run structured extraction and store the result on *host*.

        Keyword Args:
            citations (bool): When ``True`` and ``using='text'``, a shadow
                schema is sent to the LLM so it returns verbatim source
                quotes alongside each field.  The quotes are aligned back
                to PDF elements via pdfplumber's TextMap provenance data,
                producing an ``ElementCollection`` per field stored in
                ``result.citations``.  Defaults to ``False``.
        """
        schema_model = self._normalize_schema(schema)
        key = analysis_key or DEFAULT_STRUCTURED_KEY
        analyses = self._ensure_analyses(host)

        if key in analyses and not overwrite:
            logger.info(
                "Extraction for key '%s' already exists; returning cached result. "
                "Pass overwrite=True to force re-extraction.",
                key,
            )
            return analyses[key]

        citations = kwargs.pop("citations", False)
        resolved_engine = self._resolve_engine(engine, client)
        if resolved_engine == "doc_qa":
            self._perform_docqa_extraction(
                host=host,
                schema=schema_model,
                analysis_key=key,
                model=model,
                overwrite=overwrite,
                **kwargs,
            )
        else:
            self._perform_llm_extraction(
                host=host,
                schema=schema_model,
                client=client,
                analysis_key=key,
                prompt=prompt,
                using=using,
                model=model,
                overwrite=overwrite,
                citations=citations,
                **kwargs,
            )
        return analyses[key]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_analyses(host):
        if not hasattr(host, "analyses") or getattr(host, "analyses") is None:
            setattr(host, "analyses", {})
        return host.analyses

    @staticmethod
    def _normalize_schema(schema: Union[Type[BaseModel], Sequence[str]]) -> Type[BaseModel]:
        if isinstance(schema, type):
            return schema
        if not isinstance(schema, Sequence):
            raise TypeError("schema must be a Pydantic model class or a sequence of field names")
        field_names = list(schema)
        if not field_names:
            raise ValueError("Schema list cannot be empty")

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
                    alias=orig_name,
                ),
            )
        return create_model("DynamicExtractSchema", **field_defs)  # type: ignore[arg-type]

    @staticmethod
    def _resolve_engine(engine: Optional[str], client: Any) -> str:
        if engine not in (None, "llm", "doc_qa"):
            raise ValueError("engine must be either 'llm', 'doc_qa', or None")
        if engine is None:
            return "llm" if client is not None else "doc_qa"
        if engine == "llm" and client is None:
            raise ValueError("LLM engine selected but no 'client' was provided.")
        return engine

    # ------------------------------------------------------------------ #
    # Engine implementations
    # ------------------------------------------------------------------ #
    def _perform_docqa_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        analysis_key: str,
        model: Optional[str],
        overwrite: bool,
        min_confidence: float = 0.1,
        debug: bool = False,
        question_map: Optional[dict] = None,
        **kwargs,
    ) -> None:
        question_map = question_map or {}
        try:
            from pydantic import Field as _Field
            from pydantic import create_model as _create_model

            from natural_pdf.qa.document_qa import get_qa_engine
        except ImportError as exc:
            raise RuntimeError(
                'Document-QA dependencies missing. Install with `pip install "natural-pdf[qa]"`.'
            ) from exc

        qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()

        fields_iter = (
            schema.__fields__.items()
            if hasattr(schema, "__fields__")
            else schema.model_fields.items()
        )

        answers: Dict[str, Any] = {}
        confidences: Dict[str, Optional[float]] = {}
        errors: List[str] = []

        from natural_pdf.core.page import Page as _Page
        from natural_pdf.elements.region import Region as _Region

        if not isinstance(host, (_Page, _Region)):
            raise NotImplementedError(
                "Document-QA extraction is only supported on Page or Region objects."
            )

        for field_name, field_obj in fields_iter:
            display_name = getattr(field_obj, "alias", field_name)
            if display_name in question_map:
                question = question_map[display_name]
            else:
                description = None
                field_info = getattr(field_obj, "field_info", None)
                if field_info is not None and hasattr(field_info, "description"):
                    description = getattr(field_info, "description")
                elif hasattr(field_obj, "description"):
                    description = getattr(field_obj, "description")
                question = description or f"What is the {display_name.replace('_', ' ')}?"

            try:
                if isinstance(host, _Page):
                    qa_resp = qa_engine.ask_pdf_page(
                        host,
                        question,
                        min_confidence=min_confidence,
                        debug=debug,
                    )
                else:
                    qa_resp = qa_engine.ask_pdf_region(
                        host,
                        question,
                        min_confidence=min_confidence,
                        debug=debug,
                    )

                qa_item = qa_resp[0] if isinstance(qa_resp, list) and qa_resp else qa_resp

                confidence_val = None
                answer_val = None
                if isinstance(qa_item, QAResult):
                    confidence_val = qa_item.get("confidence")
                    answer_val = qa_item.get("answer")
                elif isinstance(qa_item, dict):
                    confidence_val = qa_item.get("confidence")
                    answer_val = qa_item.get("answer")

                if confidence_val is not None and confidence_val < min_confidence:
                    answer_val = None

                answers[display_name] = answer_val
                confidences[f"{display_name}_confidence"] = confidence_val
            except Exception as exc:  # noqa: BLE001
                logger.error("Doc-QA failed for field '%s': %s", field_name, exc)
                errors.append(str(exc))
                answers[display_name] = None
                confidences[f"{display_name}_confidence"] = None

        combined = {**answers, **confidences}

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

        ExtendedSchema = _create_model(f"{schema.__name__}WithConf", **field_defs_ext)

        try:
            structured_instance = ExtendedSchema(**combined)
            success_flag = not errors
            err_msg = None if not errors else "; ".join(errors)
        except Exception as exc:  # noqa: BLE001
            structured_instance = None
            success_flag = False
            err_msg = str(exc)

        result = StructuredDataResult(
            data=structured_instance,
            success=success_flag,
            error_message=err_msg,
            raw_output=combined,
            model_used=getattr(qa_engine, "model_name", None),
        )

        host.analyses[analysis_key] = result

    def _perform_llm_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str,
        prompt: Optional[str],
        using: str,
        model: Optional[str],
        overwrite: bool,
        citations: bool = False,
        **kwargs,
    ) -> None:
        if not structured_data_is_available():
            raise RuntimeError("Structured data extraction requires Pydantic; please install it.")

        if citations and using == "vision":
            logger.warning(
                "Citations are not supported with using='vision'. " "Proceeding without citations."
            )
            citations = False

        # When citations requested, get content with textmap provenance
        if citations and using == "text":
            content_getter = getattr(host, "_get_extraction_content", None)
            if callable(content_getter):
                content_result = content_getter(using=using, _return_textmap=True, **kwargs)
            else:
                content_result = self._default_extraction_content(
                    host, using=using, _return_textmap=True, **kwargs
                )

            # Unpack result based on host type
            if isinstance(content_result, tuple) and len(content_result) == 3:
                # Page/Region: (text, textmap, word_elements)
                text, textmap_info, word_elements = content_result
            elif isinstance(content_result, tuple) and len(content_result) == 2:
                # PDF: (text, list[PageTextMapInfo])
                text, textmap_info = content_result
                # Collect word_elements from all pages
                word_elements = []
                if isinstance(textmap_info, list):
                    for pinfo in textmap_info:
                        word_elements.extend(pinfo.word_elements)
            else:
                # Fallback — no textmap available
                text = content_result
                textmap_info = None
                word_elements = []

            if not text or not text.strip():
                preview = text[:120] if isinstance(text, str) else None
                msg = (
                    f"No content available for extraction (using='{using}'). "
                    "Ensure the page has a text layer or render() returns an image. "
                    "For scanned PDFs run apply_ocr() or switch to using='vision'. "
                    f"Content preview: {preview!r}"
                )
                warnings.warn(msg, RuntimeWarning)
                host.analyses[analysis_key] = StructuredDataResult(
                    data=None,
                    success=False,
                    error_message=msg,
                    raw_output=None,
                    model_used=model,
                )
                return

            from natural_pdf.extraction.citations import (
                add_line_numbers,
                build_char_to_element_map,
                build_citation_prompt,
                build_shadow_schema,
                resolve_citations,
                split_shadow_result,
            )

            # Build citation pipeline
            numbered_text, line_map = add_line_numbers(text)
            shadow_schema = build_shadow_schema(schema)
            citation_prompt = build_citation_prompt(prompt, schema)
            char_to_elem = build_char_to_element_map(word_elements)

            # Call LLM with shadow schema
            shadow_result = extract_structured_data(
                content=numbered_text,
                schema=shadow_schema,
                client=client,
                prompt=citation_prompt,
                using=using,
                model=model,
                **kwargs,
            )

            if shadow_result.success and shadow_result.data is not None:
                # Resolve citations
                citations_dict = resolve_citations(
                    shadow_data=shadow_result.data,
                    user_schema=schema,
                    line_map=line_map,
                    textmap_info=textmap_info,
                    char_to_element_map=char_to_elem,
                )
                # Split user data from shadow result
                user_data = split_shadow_result(shadow_result.data, schema)

                result = StructuredDataResult(
                    data=user_data,
                    success=True,
                    error_message=None,
                    raw_output=shadow_result.raw_output,
                    model_used=shadow_result.model_used,
                    citations=citations_dict,
                )
            else:
                result = StructuredDataResult(
                    data=shadow_result.data,
                    success=shadow_result.success,
                    error_message=shadow_result.error_message,
                    raw_output=shadow_result.raw_output,
                    model_used=shadow_result.model_used,
                )

            host.analyses[analysis_key] = result
            return

        # Standard (non-citation) path
        content_getter = getattr(host, "_get_extraction_content", None)
        if callable(content_getter):
            content = content_getter(using=using, **kwargs)
        else:
            content = self._default_extraction_content(host, using=using, **kwargs)

        if content is None or (
            using == "text" and isinstance(content, str) and not content.strip()
        ):
            preview = content[:120] if isinstance(content, str) else None
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
                raw_output=None,
                model_used=model,
            )
        else:
            result = extract_structured_data(
                content=content,
                schema=schema,
                client=client,
                prompt=prompt,
                using=using,
                model=model,
                **kwargs,
            )

        host.analyses[analysis_key] = result

    @register_delegate("extraction", "extracted")
    def extracted(
        self,
        host,
        analysis_key: Optional[str] = None,
    ) -> Optional[StructuredDataResult]:
        """Retrieve the stored :class:`StructuredDataResult` from a previous ``.extract()`` call.

        Returns the same object that ``.extract()`` returned, or ``None``
        if the extraction failed.
        """
        target_key = analysis_key if analysis_key is not None else DEFAULT_STRUCTURED_KEY

        analyses = getattr(host, "analyses", None)
        if analyses is None:
            raise AttributeError(f"{type(host).__name__} object has no 'analyses' attribute yet.")

        if target_key not in analyses:
            available_keys = list(analyses.keys())
            raise KeyError(
                f"Extraction '{target_key}' not found in analyses. " f"Available: {available_keys}"
            )

        result: StructuredDataResult = analyses[target_key]
        if not isinstance(result, StructuredDataResult):
            raise TypeError(
                f"Expected a StructuredDataResult at key '{target_key}', "
                f"found {type(result).__name__}"
            )

        if not result.success:
            logger.warning(
                "Extraction '%s' failed: %s. Returning None.",
                target_key,
                result.error_message,
            )
            return None

        return result

    def _default_extraction_content(self, host, using: str = "text", **kwargs) -> Any:
        # Pop citation-related kwarg before passing to extractors
        kwargs.pop("_return_textmap", False)
        try:
            if using == "text":
                extractor = getattr(host, "extract_text", None)
                if not callable(extractor):
                    logger.error(f"Extraction requires 'extract_text' on {host!r}")
                    return None
                layout = kwargs.pop("layout", True)
                return extractor(layout=layout, **kwargs)
            if using == "vision":
                renderer = getattr(host, "render", None)
                if not callable(renderer):
                    logger.error(f"Extraction requires 'render' on {host!r}")
                    return None
                resolution = kwargs.pop("resolution", 72)
                return renderer(resolution=resolution, **kwargs)
            logger.error(f"Unsupported value for 'using': {using}")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Error getting {using} content from {host!r}: {exc}",
                RuntimeWarning,
            )
            raise

    @staticmethod
    def _coerce_data_mapping(data: Any) -> Dict[str, Any]:
        if isinstance(data, BaseModel):
            if hasattr(data, "model_dump"):
                return data.model_dump(by_alias=True)
            return data.dict(by_alias=True)
        if isinstance(data, dict):
            return data
        if hasattr(data, "keys") and hasattr(data, "__getitem__"):
            return {key: data[key] for key in data.keys()}  # type: ignore[index]
        raise TypeError(f"Extraction returned unsupported data type {type(data).__name__}")
