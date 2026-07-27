"""Concrete public OCR capability mixins.

The method definitions in this module are the source of truth for signatures,
docstrings, IDE help, and aggregate dispatch.  OCR services continue to own
rendering, engine execution, conversion, and mutation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, Optional

from typing_extensions import Self, overload

from natural_pdf.core.ocr_contracts import (
    CustomOCRCallable,
    OCRFunctionRequest,
    OCRRequest,
    normalize_ocr_request,
)
from natural_pdf.core.ocr_execution import ocr_execution_session
from natural_pdf.ocr.replacement import OCRReplaceMode
from natural_pdf.services.base import resolve_service


class OCRScopeMixin(ABC):
    """Shared OCR contract for a direct or aggregate spatial scope."""

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[False] = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[True],
        apply_exclusions: bool = True,
        replace: Literal["ocr"] = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: None = None,
        *,
        options: None = None,
        languages: None = None,
        min_confidence: None = None,
        device: None = None,
        resolution: None = None,
        detect_only: Literal[False] = False,
        apply_exclusions: Literal[True] = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: Literal[True] = True,
        model: None = None,
        client: None = None,
        prompt: None = None,
        instructions: None = None,
        max_new_tokens: None = None,
        layout: None = None,
        preserve_markup: Literal[False] = False,
        function: CustomOCRCallable,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
    ) -> Self: ...

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: Optional[CustomOCRCallable] = None,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
    ) -> Self:
        """Apply OCR within this object's spatial scope and return ``self``.

        This method has three validated modes:

        - recognition (the default) recognizes text with a registered engine;
        - ``detect_only=True`` refreshes persistent text bounding boxes without
          deleting native or recognized text;
        - ``function=`` recognizes text with a callable receiving each physical
          Region in the scope.

        Args:
            engine: Registered OCR engine name. When omitted, resolve the
                context default. Supplying ``model`` or ``client`` selects VLM
                OCR when no engine is named.
            options: Typed engine-specific options object or validated mapping.
            languages: Ordered language codes such as ``["en", "fr"]``.
            min_confidence: Minimum accepted confidence between 0 and 1.
            device: Requested compute device, such as ``"cpu"`` or ``"cuda"``.
            resolution: Render resolution in DPI.
            detect_only: Refresh detection-only spatial artifacts instead of
                recognizing text. Detection preserves existing text.
            apply_exclusions: Mask configured exclusions in pixels sent to OCR.
            replace: Recognition/function replacement policy: ``"ocr"``,
                ``"all"``, or ``"none"``. Detection has its own refresh policy.
            use_cache: Allow the persistent OCR result cache when its identity
                can be proven safe.
            model: VLM model name.
            client: OpenAI-compatible VLM client.
            prompt: Complete VLM prompt overriding the generated prompt.
            instructions: Additional VLM instructions.
            max_new_tokens: VLM generation limit.
            layout: VLM layout mode (bool or registered detector name).
            preserve_markup: Preserve raw VLM markup in text metadata.
            function: Custom callable receiving a physical Region and returning
                recognized text or ``None``. It cannot be combined with engine,
                VLM, cache, exclusion, or detection controls.
            source_label: Provenance label stored as ``ocr_engine`` on
                custom-function output. Its selector-visible source remains
                ``"ocr"`` like every other OCR artifact.
            confidence: Confidence assigned to custom-function OCR text.

        Returns:
            The receiving object for fluent chaining.

        Raises:
            TypeError: An argument has the wrong type or ``function`` is not callable.
            ValueError: Mode-specific arguments conflict or a value is invalid.
        """

        request = normalize_ocr_request(
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
            detect_only=detect_only,
            apply_exclusions=apply_exclusions,
            replace=replace,
            use_cache=use_cache,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
            function=function,
            source_label=source_label,
            confidence=confidence,
        )
        with ocr_execution_session():
            self._execute_ocr_request(request)
            self._after_apply_ocr(request)
        return self

    def _execute_ocr_request(self, request: OCRRequest) -> None:
        hosts = tuple(self._iter_ocr_hosts(request))
        for host in hosts:
            execute = getattr(host, "_execute_ocr_request", None)
            if not callable(execute):
                raise TypeError(
                    f"{host.__class__.__name__} is not a concrete OCR target; "
                    "aggregate OCR hooks must yield OCR-capable hosts"
                )
            execute(request)

    @abstractmethod
    def _iter_ocr_hosts(self, request: OCRRequest) -> Iterable[Any]:
        """Yield concrete OCR-capable hosts in deterministic scope order."""

    def _after_apply_ocr(self, request: OCRRequest) -> None:
        """Run a host-specific post-success hook."""


class OCRDirectTargetMixin(OCRScopeMixin):
    """Protected execution hooks for Page- and Region-like physical targets."""

    def _iter_ocr_hosts(self, request: OCRRequest) -> Iterable[Any]:
        yield self

    def _before_apply_ocr(self, request: OCRRequest) -> None:
        """Run target-specific validation before OCR."""

    def _ocr_function_target(self) -> Any:
        """Return the physical Region passed to a custom OCR function."""

        return self

    def _execute_ocr_request(self, request: OCRRequest) -> None:
        self._before_apply_ocr(request)
        service = resolve_service(self, "ocr")
        if isinstance(request, OCRFunctionRequest):
            service.apply_custom_ocr(
                self._ocr_function_target(),
                ocr_function=request.function,
                source_label=request.source_label,
                confidence=request.confidence,
                replace=request.replace,
                add_to_page=True,
            )
            return
        service.apply_ocr(self, **request.service_kwargs())


class PDFOCRMixin(ABC):
    """Shared public OCR contract for one PDF with selectable pages."""

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[False] = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        show_progress: bool = True,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[True],
        apply_exclusions: bool = True,
        replace: Literal["ocr"] = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        show_progress: bool = True,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: None = None,
        *,
        options: None = None,
        languages: None = None,
        min_confidence: None = None,
        device: None = None,
        resolution: None = None,
        detect_only: Literal[False] = False,
        apply_exclusions: Literal[True] = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: Literal[True] = True,
        model: None = None,
        client: None = None,
        prompt: None = None,
        instructions: None = None,
        max_new_tokens: None = None,
        layout: None = None,
        preserve_markup: Literal[False] = False,
        function: CustomOCRCallable,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        show_progress: bool = True,
    ) -> Self: ...

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: Optional[CustomOCRCallable] = None,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        show_progress: bool = True,
    ) -> Self:
        """Apply OCR to selected pages and return ``self``.

        This method has three validated modes:

        - recognition (the default) recognizes text with a registered engine;
        - ``detect_only=True`` refreshes persistent text bounding boxes without
          deleting native or recognized text;
        - ``function=`` recognizes text with a callable receiving each physical
          Region on the selected pages.

        Args:
            engine: Registered OCR engine name. When omitted, resolve the PDF
                context default. Supplying ``model`` or ``client`` selects VLM
                OCR when no engine is named.
            options: Typed engine-specific options object or validated mapping.
            languages: Ordered language codes such as ``["en", "fr"]``.
            min_confidence: Minimum accepted confidence between 0 and 1.
            device: Requested compute device, such as ``"cpu"`` or ``"cuda"``.
            resolution: Render resolution in DPI.
            detect_only: Refresh detection-only spatial artifacts instead of
                recognizing text. Detection preserves existing text.
            apply_exclusions: Mask configured exclusions in pixels sent to OCR.
            replace: Recognition/function replacement policy: ``"ocr"``,
                ``"all"``, or ``"none"``. Detection has its own refresh policy.
            use_cache: Allow the persistent OCR result cache when its identity
                can be proven safe.
            model: VLM model name.
            client: OpenAI-compatible VLM client.
            prompt: Complete VLM prompt overriding the generated prompt.
            instructions: Additional VLM instructions.
            max_new_tokens: VLM generation limit.
            layout: VLM layout mode (bool or registered detector name).
            preserve_markup: Preserve raw VLM markup in text metadata.
            function: Custom callable receiving a physical Region and returning
                recognized text or ``None``. It cannot be combined with engine,
                VLM, cache, exclusion, or detection controls.
            source_label: Provenance label stored as ``ocr_engine`` on
                custom-function output. Its selector-visible source remains
                ``"ocr"`` like every other OCR artifact.
            confidence: Confidence assigned to custom-function OCR text.
            pages: Page index, iterable of indexes, range, or slice to process.
                Omit to process every page in PDF order.
            show_progress: Display a per-page progress bar while executing.

        Returns:
            The PDF for fluent chaining.

        Raises:
            TypeError: An argument has the wrong type or ``function`` is not callable.
            ValueError: Mode-specific arguments conflict or a value is invalid.
        """

        if not isinstance(show_progress, bool):
            raise TypeError("show_progress must be a bool")
        request = normalize_ocr_request(
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
            detect_only=detect_only,
            apply_exclusions=apply_exclusions,
            replace=replace,
            use_cache=use_cache,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
            function=function,
            source_label=source_label,
            confidence=confidence,
        )
        with ocr_execution_session():
            self._apply_pdf_ocr_request(request, pages=pages, show_progress=show_progress)
        return self

    @abstractmethod
    def _apply_pdf_ocr_request(
        self,
        request: OCRRequest,
        *,
        pages: Optional[int | Iterable[int] | range | slice],
        show_progress: bool,
    ) -> None: ...


class PDFCollectionOCRMixin(ABC):
    """Shared public OCR contract for a collection of PDFs."""

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[False] = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: Literal[True],
        apply_exclusions: bool = True,
        replace: Literal["ocr"] = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: None = None,
        source_label: Literal["custom-ocr"] = "custom-ocr",
        confidence: None = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> Self: ...

    @overload
    def apply_ocr(
        self,
        engine: None = None,
        *,
        options: None = None,
        languages: None = None,
        min_confidence: None = None,
        device: None = None,
        resolution: None = None,
        detect_only: Literal[False] = False,
        apply_exclusions: Literal[True] = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: Literal[True] = True,
        model: None = None,
        client: None = None,
        prompt: None = None,
        instructions: None = None,
        max_new_tokens: None = None,
        layout: None = None,
        preserve_markup: Literal[False] = False,
        function: CustomOCRCallable,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> Self: ...

    def apply_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[list[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: Optional[CustomOCRCallable] = None,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
        pages: Optional[int | Iterable[int] | range | slice] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> Self:
        """Apply OCR across PDFs and return ``self``.

        This method has three validated modes:

        - recognition (the default) recognizes text with a registered engine;
        - ``detect_only=True`` refreshes persistent text bounding boxes without
          deleting native or recognized text;
        - ``function=`` recognizes text with a callable receiving each physical
          Region on the selected pages of every PDF.

        Args:
            engine: Registered OCR engine name. When omitted, resolve each
                PDF's context default. Supplying ``model`` or ``client`` selects
                VLM OCR when no engine is named.
            options: Typed engine-specific options object or validated mapping.
            languages: Ordered language codes such as ``["en", "fr"]``.
            min_confidence: Minimum accepted confidence between 0 and 1.
            device: Requested compute device, such as ``"cpu"`` or ``"cuda"``.
            resolution: Render resolution in DPI.
            detect_only: Refresh detection-only spatial artifacts instead of
                recognizing text. Detection preserves existing text.
            apply_exclusions: Mask configured exclusions in pixels sent to OCR.
            replace: Recognition/function replacement policy: ``"ocr"``,
                ``"all"``, or ``"none"``. Detection has its own refresh policy.
            use_cache: Allow the persistent OCR result cache when its identity
                can be proven safe.
            model: VLM model name.
            client: OpenAI-compatible VLM client.
            prompt: Complete VLM prompt overriding the generated prompt.
            instructions: Additional VLM instructions.
            max_new_tokens: VLM generation limit.
            layout: VLM layout mode (bool or registered detector name).
            preserve_markup: Preserve raw VLM markup in text metadata.
            function: Custom callable receiving a physical Region and returning
                recognized text or ``None``. It cannot be combined with engine,
                VLM, cache, exclusion, or detection controls.
            source_label: Provenance label stored as ``ocr_engine`` on
                custom-function output. Its selector-visible source remains
                ``"ocr"`` like every other OCR artifact.
            confidence: Confidence assigned to custom-function OCR text.
            pages: Page index, iterable of indexes, range, or slice to process
                for every PDF. Omit to process every page in PDF order.
            max_workers: Maximum PDFs to process concurrently. ``None`` uses
                the collection default; ``1`` runs serially.
            show_progress: Display a collection-level progress bar while PDFs
                complete. Individual PDF progress is suppressed to avoid nested
                bars.

        Returns:
            The PDF collection for fluent chaining.

        Raises:
            TypeError: An argument has the wrong type or ``function`` is not callable.
            ValueError: Mode-specific arguments conflict or a value is invalid.
        """

        if not isinstance(show_progress, bool):
            raise TypeError("show_progress must be a bool")
        if max_workers is not None and (
            isinstance(max_workers, bool) or not isinstance(max_workers, int)
        ):
            raise TypeError("max_workers must be a positive integer or None")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be a positive integer or None")
        request = normalize_ocr_request(
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
            detect_only=detect_only,
            apply_exclusions=apply_exclusions,
            replace=replace,
            use_cache=use_cache,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
            function=function,
            source_label=source_label,
            confidence=confidence,
        )
        with ocr_execution_session():
            self._apply_pdf_collection_ocr_request(
                request,
                pages=pages,
                max_workers=max_workers,
                show_progress=show_progress,
            )
        return self

    @abstractmethod
    def _apply_pdf_collection_ocr_request(
        self,
        request: OCRRequest,
        *,
        pages: Optional[int | Iterable[int] | range | slice],
        max_workers: Optional[int],
        show_progress: bool,
    ) -> None: ...


__all__ = [
    "OCRDirectTargetMixin",
    "OCRScopeMixin",
    "PDFCollectionOCRMixin",
    "PDFOCRMixin",
]
