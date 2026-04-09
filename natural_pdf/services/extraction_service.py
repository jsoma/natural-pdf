from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

from pydantic import BaseModel, Field, create_model

from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.extraction.structured_ops import (
    extract_structured_data,
    structured_data_is_available,
)
from natural_pdf.services._model_support import (
    DOC_QA_INSTALL_MESSAGE,
    VISION_MODE_REQUIREMENTS,
    VLM_INSTALL_MESSAGE,
)
from natural_pdf.services.registry import register_delegate

DEFAULT_STRUCTURED_KEY = "structured"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedExtractionMode:
    using: str
    engine: str


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
            citations (bool): When ``True`` and ``using='text'``, source
                line numbers are collected alongside each field and resolved
                to PDF elements via pdfplumber's TextMap provenance data,
                producing an ``ElementCollection`` per field stored in
                ``result.citations``.  Defaults to ``False``.
            confidence: Per-field confidence scoring. Accepts ``True`` or
                ``'range'`` for default 1–5 numeric scale, a ``list``
                for categorical levels, or a ``dict`` mapping values to
                descriptions.  Results are available via
                ``result["field"].confidence`` and ``result.confidences``.
                Defaults to ``None`` (disabled).
            instructions (str): Domain-specific guidance appended to the
                LLM prompt.  Affects all reasoning (extraction, citations,
                confidence).  Defaults to ``None``.
            multipass (bool): When ``True``, uses two LLM calls (extraction
                then combined meta for citations+confidence) instead of the
                default single-pass approach.  Defaults to ``False``.
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
        confidence = kwargs.pop("confidence", None)
        instructions = kwargs.pop("instructions", None)
        multipass = kwargs.pop("multipass", False)
        resolved_mode = self._resolve_mode(using=using, engine=engine, client=client)
        if resolved_mode.engine == "doc_qa":
            self._perform_docqa_extraction(
                host=host,
                schema=schema_model,
                analysis_key=key,
                model=model,
                overwrite=overwrite,
                **kwargs,
            )
        elif resolved_mode.engine == "vlm":
            self._perform_vlm_extraction(
                host=host,
                schema=schema_model,
                analysis_key=key,
                prompt=prompt,
                model=model,
                **kwargs,
            )
        else:
            self._perform_llm_extraction(
                host=host,
                schema=schema_model,
                client=client,
                analysis_key=key,
                prompt=prompt,
                using=resolved_mode.using,
                model=model,
                overwrite=overwrite,
                citations=citations,
                confidence=confidence,
                instructions=instructions,
                multipass=multipass,
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
    def _resolve_mode(*, using: str, engine: Optional[str], client: Any) -> ResolvedExtractionMode:
        normalized_using = (using or "text").strip().lower()
        if normalized_using not in {"text", "vision"}:
            raise ValueError("using must be 'text' or 'vision'")

        if engine == "vlm":
            return ResolvedExtractionMode(using="vision", engine="vlm")
        if engine not in (None, "llm", "doc_qa"):
            raise ValueError("engine must be 'llm', 'doc_qa', 'vlm', or None")
        if normalized_using == "vision" and client is None and engine != "vlm":
            raise ValueError(VISION_MODE_REQUIREMENTS)
        if engine is None:
            resolved_engine = "llm" if client is not None else "doc_qa"
            return ResolvedExtractionMode(using=normalized_using, engine=resolved_engine)
        if engine == "llm" and client is None:
            raise ValueError("LLM engine selected but no 'client' was provided.")
        return ResolvedExtractionMode(using=normalized_using, engine=engine)

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
            raise RuntimeError(DOC_QA_INSTALL_MESSAGE) from exc

        qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()

        fields_iter = (
            schema.model_fields.items()
            if hasattr(schema, "model_fields")
            else schema.__fields__.items()
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
            display_name = getattr(field_obj, "alias", None) or field_name
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
                if isinstance(qa_item, dict):
                    confidence_val = qa_item.get("confidence")
                    answer_val = qa_item.get("answer")
                elif hasattr(qa_item, "get"):
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

    def _perform_vlm_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        analysis_key: str,
        prompt: Optional[str],
        model: Optional[str],
        **kwargs,
    ) -> None:
        """Run extraction using a local HuggingFace VLM."""
        try:
            from natural_pdf.extraction.vlm_adapter import get_vlm_adapter
        except ImportError as exc:
            raise RuntimeError(VLM_INSTALL_MESSAGE) from exc

        # Get image from host
        renderer = getattr(host, "render", None)
        if not callable(renderer):
            raise RuntimeError(f"VLM extraction requires 'render' on {host!r}")
        resolution = kwargs.pop("resolution", 150)
        image = renderer(resolution=resolution)

        adapter = get_vlm_adapter(model_name=model)
        effective_prompt = prompt or (
            f"Extract the information corresponding to the fields in the "
            f"{schema.__name__} schema from this document image."
        )

        from natural_pdf.core.vlm_client import DEFAULT_VLM_MAX_TOKENS

        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_VLM_MAX_TOKENS)

        try:
            parsed = adapter.generate(
                image=image,
                prompt=effective_prompt,
                schema=schema,
                max_new_tokens=max_new_tokens,
            )
            result = StructuredDataResult(
                data=parsed,
                success=True,
                error_message=None,
                raw_output=None,
                model_used=adapter.model_name,
            )
        except Exception as exc:
            result = StructuredDataResult(
                data=None,
                success=False,
                error_message=str(exc),
                raw_output=None,
                model_used=adapter.model_name,
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
        confidence=None,
        instructions: Optional[str] = None,
        multipass: bool = False,
        **kwargs,
    ) -> None:
        if not structured_data_is_available():
            raise RuntimeError("Structured data extraction requires Pydantic; please install it.")

        from natural_pdf.extraction.citations import (
            add_line_numbers,
            build_char_to_element_map,
            build_extended_prompt,
            build_extended_schema,
            build_meta_prompt,
            build_meta_schema,
            normalize_confidence_config,
            resolve_citations,
            resolve_source_lines_to_text,
            split_extended_result,
        )

        # Normalize confidence config
        confidence_config = normalize_confidence_config(confidence)
        use_confidence = confidence_config is not None
        use_citations = bool(citations)

        if use_citations and using == "vision":
            logger.warning(
                "Citations are not supported with using='vision'. Proceeding without citations."
            )
            use_citations = False

        # Citations require textmap provenance (text mode only)
        need_textmap = use_citations and using == "text"
        need_meta = use_citations or use_confidence

        # ---- Get content ---- #
        if need_textmap:
            content_getter = getattr(host, "_get_extraction_content", None)
            if callable(content_getter):
                content_result = content_getter(using=using, _return_textmap=True, **kwargs)
            else:
                content_result = self._default_extraction_content(
                    host, using=using, _return_textmap=True, **kwargs
                )

            if isinstance(content_result, tuple) and len(content_result) == 3:
                text, textmap_info, word_elements = content_result
            elif isinstance(content_result, tuple) and len(content_result) == 2:
                text, textmap_info = content_result
                word_elements = []
                if isinstance(textmap_info, list):
                    for pinfo in textmap_info:
                        word_elements.extend(pinfo.word_elements)
            else:
                text = content_result
                textmap_info = None
                word_elements = []
            content = text
        else:
            content_getter = getattr(host, "_get_extraction_content", None)
            if callable(content_getter):
                content = content_getter(using=using, **kwargs)
            else:
                content = self._default_extraction_content(host, using=using, **kwargs)

        # Empty-content guard
        if content is None or (
            using == "text" and isinstance(content, str) and not content.strip()
        ):
            msg = (
                "No text content found for extraction. "
                "This PDF may be scanned/image-based. Try:\n"
                "  - using='vision' to send page images to the LLM\n"
                "  - page.apply_ocr() first to add a text layer\n"
                "\n"
                "Currently using='text' (the default), which only works "
                "when the PDF has embedded text."
            )
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning)
            result = StructuredDataResult(
                data=None,
                success=False,
                error_message=msg,
                raw_output=None,
                model_used=model,
            )
            host.analyses[analysis_key] = result
            return result

        # Build numbered text if citations are needed
        numbered_text = None
        line_map = None
        if use_citations and using == "text" and isinstance(content, str):
            numbered_text, line_map = add_line_numbers(content)

        # ---- Branch on multipass vs single-pass ---- #
        if not need_meta:
            # No meta requested — plain extraction
            self._plain_extraction(
                host=host,
                schema=schema,
                client=client,
                analysis_key=analysis_key,
                prompt=prompt,
                using=using,
                model=model,
                content=content,
                instructions=instructions,
                **kwargs,
            )
        elif multipass:
            # Two-pass: extract data, then combined meta
            self._two_pass_extraction(
                host=host,
                schema=schema,
                client=client,
                analysis_key=analysis_key,
                prompt=prompt,
                using=using,
                model=model,
                content=content,
                instructions=instructions,
                use_citations=use_citations,
                use_confidence=use_confidence,
                confidence_config=confidence_config,
                need_textmap=need_textmap,
                numbered_text=numbered_text,
                line_map=line_map,
                textmap_info=textmap_info if need_textmap else None,
                word_elements=word_elements if need_textmap else [],
                **kwargs,
            )
        else:
            # Single-pass: combined schema with user fields + meta
            self._single_pass_extraction(
                host=host,
                schema=schema,
                client=client,
                analysis_key=analysis_key,
                prompt=prompt,
                using=using,
                model=model,
                content=content,
                instructions=instructions,
                use_citations=use_citations,
                use_confidence=use_confidence,
                confidence_config=confidence_config,
                need_textmap=need_textmap,
                numbered_text=numbered_text,
                line_map=line_map,
                textmap_info=textmap_info if need_textmap else None,
                word_elements=word_elements if need_textmap else [],
                **kwargs,
            )

    def _plain_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str,
        prompt: Optional[str],
        using: str,
        model: Optional[str],
        content: Any,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plain extraction with no citations or confidence."""
        effective_prompt = prompt
        if instructions:
            base = prompt or (
                f"Extract the information corresponding to the fields in the "
                f"{schema.__name__} schema. Respond only with the structured data."
            )
            effective_prompt = f"{base}\n\n{instructions}"

        pass1_result = extract_structured_data(
            content=content,
            schema=schema,
            client=client,
            prompt=effective_prompt,
            using=using,
            model=model,
            **kwargs,
        )

        host.analyses[analysis_key] = StructuredDataResult(
            data=pass1_result.data,
            success=pass1_result.success,
            error_message=pass1_result.error_message,
            raw_output=pass1_result.raw_output,
            model_used=pass1_result.model_used,
        )

    def _single_pass_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str,
        prompt: Optional[str],
        using: str,
        model: Optional[str],
        content: Any,
        instructions: Optional[str] = None,
        use_citations: bool = False,
        use_confidence: bool = False,
        confidence_config=None,
        need_textmap: bool = False,
        numbered_text: Optional[str] = None,
        line_map: Optional[Dict[int, str]] = None,
        textmap_info: Any = None,
        word_elements: list = None,
        **kwargs,
    ) -> None:
        """Single LLM call with combined schema (user fields + source_lines + confidence)."""
        from natural_pdf.extraction.citations import (
            build_char_to_element_map,
            build_extended_prompt,
            build_extended_schema,
            resolve_citations,
            resolve_source_lines_to_text,
            split_extended_result,
        )

        if word_elements is None:
            word_elements = []

        # Build extended schema with source_lines and/or confidence
        extended_schema = build_extended_schema(
            schema,
            with_sources=use_citations,
            with_confidence=use_confidence,
            confidence_config=confidence_config,
        )

        # Build extended prompt
        extended_prompt = build_extended_prompt(
            prompt,
            schema,
            instructions=instructions,
            with_sources=use_citations,
            with_confidence=use_confidence,
            confidence_config=confidence_config,
        )

        # Use numbered text as content if citations requested
        llm_content = numbered_text if (use_citations and numbered_text) else content

        result = extract_structured_data(
            content=llm_content,
            schema=extended_schema,
            client=client,
            prompt=extended_prompt,
            using=using,
            model=model,
            **kwargs,
        )

        if not result.success or result.data is None:
            host.analyses[analysis_key] = StructuredDataResult(
                data=result.data,
                success=result.success,
                error_message=result.error_message,
                raw_output=result.raw_output,
                model_used=result.model_used,
            )
            return

        # Split the extended result
        data_dict = (
            result.data.model_dump() if hasattr(result.data, "model_dump") else result.data.dict()
        )
        user_data, sources_dict, confidences_dict = split_extended_result(
            data_dict, schema, with_sources=use_citations, with_confidence=use_confidence
        )

        # Re-create user model instance (model_construct bypasses alias
        # validation so field-name keys work even when aliases are set)
        user_model_instance = schema.model_construct(**user_data)

        # Resolve citations to ElementCollections
        citations_dict = None
        resolved_sources = None
        if use_citations and sources_dict and line_map:
            # Resolve line numbers to text strings for FieldResult.sources
            resolved_sources = resolve_source_lines_to_text(sources_dict, line_map)

            if need_textmap:
                char_to_elem = build_char_to_element_map(word_elements)
                citations_dict = resolve_citations(
                    shadow_data=data_dict,
                    user_schema=schema,
                    line_map=line_map,
                    textmap_info=textmap_info,
                    char_to_element_map=char_to_elem,
                )

        # Clamp confidences
        if confidences_dict:
            confidences_dict = self._clamp_confidences(confidences_dict, confidence_config)

        host.analyses[analysis_key] = StructuredDataResult(
            data=user_model_instance,
            success=True,
            error_message=None,
            raw_output=result.raw_output,
            model_used=result.model_used,
            citations=citations_dict,
            confidences=confidences_dict,
            sources=resolved_sources,
        )

    def _two_pass_extraction(
        self,
        *,
        host,
        schema: Type[BaseModel],
        client: Any,
        analysis_key: str,
        prompt: Optional[str],
        using: str,
        model: Optional[str],
        content: Any,
        instructions: Optional[str] = None,
        use_citations: bool = False,
        use_confidence: bool = False,
        confidence_config=None,
        need_textmap: bool = False,
        numbered_text: Optional[str] = None,
        line_map: Optional[Dict[int, str]] = None,
        textmap_info: Any = None,
        word_elements: list = None,
        **kwargs,
    ) -> None:
        """Two LLM calls: extract data, then combined meta (citations+confidence)."""
        from natural_pdf.extraction.citations import (
            build_char_to_element_map,
            build_meta_prompt,
            build_meta_schema,
            resolve_citations,
            resolve_source_lines_to_text,
        )

        if word_elements is None:
            word_elements = []

        # ---- Pass 1: Plain extraction ---- #
        effective_prompt = prompt
        if instructions:
            base = prompt or (
                f"Extract the information corresponding to the fields in the "
                f"{schema.__name__} schema. Respond only with the structured data."
            )
            effective_prompt = f"{base}\n\n{instructions}"

        pass1_result = extract_structured_data(
            content=content,
            schema=schema,
            client=client,
            prompt=effective_prompt,
            using=using,
            model=model,
            **kwargs,
        )

        if not pass1_result.success or pass1_result.data is None:
            host.analyses[analysis_key] = StructuredDataResult(
                data=pass1_result.data,
                success=pass1_result.success,
                error_message=pass1_result.error_message,
                raw_output=pass1_result.raw_output,
                model_used=pass1_result.model_used,
            )
            return

        user_data_dict = (
            pass1_result.data.model_dump()
            if hasattr(pass1_result.data, "model_dump")
            else pass1_result.data.dict()
        )

        # ---- Pass 2: Combined meta (citations + confidence) ---- #
        meta_schema = build_meta_schema(
            schema,
            with_sources=use_citations,
            with_confidence=use_confidence,
            confidence_config=confidence_config,
        )
        meta_prompt = build_meta_prompt(
            schema,
            user_data_dict,
            with_sources=use_citations,
            with_confidence=use_confidence,
            confidence_config=confidence_config,
        )

        # Use numbered text as content for pass 2 if citations requested
        llm_content = numbered_text if (use_citations and numbered_text) else content

        sources_dict = None
        citations_dict = None
        confidences_dict = None
        resolved_sources = None

        try:
            meta_result = extract_structured_data(
                content=llm_content,
                schema=meta_schema,
                client=client,
                prompt=meta_prompt,
                using=using,
                model=model,
                **kwargs,
            )
            if meta_result.success and meta_result.data is not None:
                meta_dict = (
                    meta_result.data.model_dump()
                    if hasattr(meta_result.data, "model_dump")
                    else meta_result.data.dict()
                )

                # Extract sources (strip _source_lines suffix)
                if use_citations:
                    sources_dict = {}
                    for key, val in meta_dict.items():
                        if key.endswith("_source_lines"):
                            field_name = key[: -len("_source_lines")]
                            sources_dict[field_name] = val

                    # Resolve line numbers to text strings
                    if line_map:
                        resolved_sources = resolve_source_lines_to_text(sources_dict, line_map)

                    # Resolve to ElementCollections
                    if need_textmap and line_map:
                        char_to_elem = build_char_to_element_map(word_elements)
                        citations_dict = resolve_citations(
                            shadow_data=meta_dict,
                            user_schema=schema,
                            line_map=line_map,
                            textmap_info=textmap_info,
                            char_to_element_map=char_to_elem,
                        )

                # Extract confidences (strip _confidence suffix)
                if use_confidence:
                    confidences_dict = {}
                    for key, val in meta_dict.items():
                        if key.endswith("_confidence"):
                            field_name = key[: -len("_confidence")]
                            confidences_dict[field_name] = val
                    confidences_dict = self._clamp_confidences(confidences_dict, confidence_config)
        except Exception:
            logger.warning(
                "Meta pass failed; returning extraction without meta.",
                exc_info=True,
            )
            # Graceful degradation for confidence
            if use_confidence:
                if hasattr(schema, "model_fields"):
                    field_names = list(schema.model_fields.keys())
                else:
                    field_names = list(schema.__fields__.keys())
                confidences_dict = {fname: None for fname in field_names}

        host.analyses[analysis_key] = StructuredDataResult(
            data=pass1_result.data,
            success=True,
            error_message=None,
            raw_output=pass1_result.raw_output,
            model_used=pass1_result.model_used,
            citations=citations_dict,
            confidences=confidences_dict,
            sources=resolved_sources,
        )

    @staticmethod
    def _clamp_confidences(
        confidences: Dict[str, Any],
        config: Any,
    ) -> Dict[str, Any]:
        """Clamp numeric confidence values to the configured scale bounds."""
        if config is None or not config.is_numeric:
            return confidences
        clamped = {}
        for field_name, val in confidences.items():
            if val is None:
                clamped[field_name] = None
            elif isinstance(val, (int, float)):
                if val < config.min_value:
                    logger.warning(
                        "Confidence for '%s' was %s, clamping to %s.",
                        field_name,
                        val,
                        config.min_value,
                    )
                    clamped[field_name] = config.min_value
                elif val > config.max_value:
                    logger.warning(
                        "Confidence for '%s' was %s, clamping to %s.",
                        field_name,
                        val,
                        config.max_value,
                    )
                    clamped[field_name] = config.max_value
                else:
                    clamped[field_name] = val
            else:
                clamped[field_name] = val
        return clamped

    @register_delegate("extraction", "extracted")
    def extracted(
        self,
        host,
        analysis_key: Optional[str] = None,
    ) -> StructuredDataResult:
        """Retrieve the stored :class:`StructuredDataResult` from a previous ``.extract()`` call.

        Always returns the result object. Check ``.success`` to see if
        extraction succeeded and ``.error_message`` for failure details.
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
                "Extraction '%s' failed: %s",
                target_key,
                result.error_message,
            )

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
