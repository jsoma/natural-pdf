"""Public text-extraction signature families.

Each public signature and docstring lives here exactly once.  Concrete hosts
implement only the small acquisition hooks documented by their mixin.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from natural_pdf.text.contracts import (
    AggregatePolicy,
    ContentFilter,
    ExtractedText,
    TextLayoutOptions,
    WhitespaceMode,
)
from natural_pdf.text.pipeline import (
    apply_prepared_text_transform,
    join_extracted_text,
    prepare_text_transform,
    scalar_text_result,
    validate_layout_request,
)


class SpatialTextMixin:
    """Text facade for Page, Region, and other spatial hosts."""

    def _extract_spatial_text_result(
        self,
        *,
        layout: bool | TextLayoutOptions,
        apply_exclusions: bool,
    ) -> ExtractedText:
        """Acquire raw spatial text; concrete spatial hosts must implement this hook."""

        raise NotImplementedError

    def extract_text(
        self,
        *,
        layout: bool | TextLayoutOptions = False,
        apply_exclusions: bool = True,
        newlines: bool | str = True,
        whitespace: WhitespaceMode = "preserve",
        strip: bool = True,
        bidi: bool = True,
        content_filter: ContentFilter | None = None,
    ) -> str:
        """Extract spatial text with explicit acquisition and transform options.

        ``layout`` enables spatial layout reconstruction, while
        ``apply_exclusions`` controls registered exclusion regions. Newline,
        whitespace, bidi, filtering, and stripping transforms are applied in a
        stable order after acquisition. Regex filters remove matches; callable
        filters are predicates invoked once for each Unicode codepoint.
        """

        validate_layout_request(layout)
        if not isinstance(apply_exclusions, bool):
            raise TypeError("apply_exclusions must be a bool")
        request = prepare_text_transform(
            newlines=newlines,
            whitespace=whitespace,
            strip=strip,
            bidi=bidi,
            content_filter=content_filter,
        )
        result = self._extract_spatial_text_result(
            layout=layout,
            apply_exclusions=apply_exclusions,
        )
        if not isinstance(result, ExtractedText):
            raise TypeError("_extract_spatial_text_result must return ExtractedText")
        return apply_prepared_text_transform(result.text, request)

    def extract_text_result(
        self,
        *,
        layout: bool | TextLayoutOptions = False,
        apply_exclusions: bool = True,
    ) -> ExtractedText:
        """Return raw spatial text and provenance using acquisition options only."""

        validate_layout_request(layout)
        if not isinstance(apply_exclusions, bool):
            raise TypeError("apply_exclusions must be a bool")
        result = self._extract_spatial_text_result(
            layout=layout,
            apply_exclusions=apply_exclusions,
        )
        if not isinstance(result, ExtractedText):
            raise TypeError("_extract_spatial_text_result must return ExtractedText")
        return result


class ScalarTextMixin:
    """Text facade for a single scalar text value."""

    def _extract_scalar_text(self) -> str:
        """Acquire raw scalar text; concrete scalar hosts must implement this hook."""

        raise NotImplementedError

    def extract_text(
        self,
        *,
        newlines: bool | str = True,
        whitespace: WhitespaceMode = "preserve",
        strip: bool = True,
        content_filter: ContentFilter | None = None,
    ) -> str:
        """Extract scalar text and apply newline, whitespace, filter, and strip transforms."""

        request = prepare_text_transform(
            newlines=newlines,
            whitespace=whitespace,
            strip=strip,
            bidi=False,
            content_filter=content_filter,
        )
        return apply_prepared_text_transform(self._extract_scalar_text(), request)

    def extract_text_result(self) -> ExtractedText:
        """Return the raw scalar value with its source span."""

        return scalar_text_result(self._extract_scalar_text(), source=self)


class AggregateTextMixin:
    """Text facade for ordered page-, document-, or section-like members."""

    def _iter_text_members(self) -> Iterable[Any]:
        """Yield ordered aggregate members; concrete aggregate hosts implement this hook."""

        raise NotImplementedError

    def _text_aggregate_policy(self) -> AggregatePolicy:
        """Return this host's natural separator and empty-member policy."""

        raise NotImplementedError

    def extract_text(
        self,
        *,
        separator: str | None = None,
        layout: bool | TextLayoutOptions = False,
        apply_exclusions: bool = True,
        newlines: bool | str = True,
        whitespace: WhitespaceMode = "preserve",
        strip: bool = True,
        bidi: bool = True,
        content_filter: ContentFilter | None = None,
    ) -> str:
        """Extract members independently, then join them at exact host boundaries.

        ``separator=None`` uses the host's natural separator. Empty member
        handling is host policy. Transforms run on members only: separators are
        never normalized, stripped, bidi-processed, or included in a regex match.
        """

        validate_layout_request(layout)
        if separator is not None and not isinstance(separator, str):
            raise TypeError("separator must be a str or None")
        if not isinstance(apply_exclusions, bool):
            raise TypeError("apply_exclusions must be a bool")
        # Validate before asking the host for its policy or members. Members'
        # public facades enforce the same leaf contract when invoked below.
        prepare_text_transform(
            newlines=newlines,
            whitespace=whitespace,
            strip=strip,
            bidi=bidi,
            content_filter=content_filter,
        )
        policy = self._text_aggregate_policy()
        if not isinstance(policy, AggregatePolicy):
            raise TypeError("_text_aggregate_policy must return AggregatePolicy")
        actual_separator = policy.natural_separator if separator is None else separator

        texts: list[str] = []
        for member in self._iter_text_members():
            extractor = getattr(member, "extract_text", None)
            if not callable(extractor):
                raise TypeError(f"aggregate member {member!r} has no extract_text method")
            text = extractor(
                layout=layout,
                apply_exclusions=apply_exclusions,
                newlines=newlines,
                whitespace=whitespace,
                strip=strip,
                bidi=bidi,
                content_filter=content_filter,
            )
            if not isinstance(text, str):
                raise TypeError("aggregate member extract_text must return str")
            if policy.preserve_empty or text:
                texts.append(text)
        return actual_separator.join(texts)

    def extract_text_result(
        self,
        *,
        separator: str | None = None,
        layout: bool | TextLayoutOptions = False,
        apply_exclusions: bool = True,
    ) -> ExtractedText:
        """Join raw member results with exact source offsets."""

        validate_layout_request(layout)
        if separator is not None and not isinstance(separator, str):
            raise TypeError("separator must be a str or None")
        if not isinstance(apply_exclusions, bool):
            raise TypeError("apply_exclusions must be a bool")
        policy = self._text_aggregate_policy()
        if not isinstance(policy, AggregatePolicy):
            raise TypeError("_text_aggregate_policy must return AggregatePolicy")
        actual_separator = policy.natural_separator if separator is None else separator

        results: list[ExtractedText] = []
        for member in self._iter_text_members():
            result_extractor = getattr(member, "extract_text_result", None)
            if callable(result_extractor):
                result = result_extractor(
                    layout=layout,
                    apply_exclusions=apply_exclusions,
                )
            else:
                extractor = getattr(member, "extract_text", None)
                if not callable(extractor):
                    raise TypeError(f"aggregate member {member!r} has no text extraction method")
                result = scalar_text_result(extractor(), source=member)
            if not isinstance(result, ExtractedText):
                raise TypeError("aggregate member extract_text_result must return ExtractedText")
            results.append(result)
        return join_extracted_text(
            results,
            separator=actual_separator,
            preserve_empty=policy.preserve_empty,
        )


class SelectedTextMixin:
    """Literal, ordered text facade for selected heterogeneous elements."""

    def _iter_selected_text_members(self) -> Iterable[Any]:
        """Yield selected members in stored order, including duplicates."""

        raise NotImplementedError

    @staticmethod
    def _selected_raw_result(member: Any) -> ExtractedText | None:
        if isinstance(member, SpatialTextMixin):
            return member._extract_spatial_text_result(layout=False, apply_exclusions=True)
        if isinstance(member, ScalarTextMixin):
            return scalar_text_result(member._extract_scalar_text(), source=member)
        result_extractor = getattr(member, "extract_text_result", None)
        if callable(result_extractor):
            result = result_extractor()
            if isinstance(result, ExtractedText):
                return result
        extractor = getattr(member, "extract_text", None)
        if not callable(extractor):
            return None
        text = extractor()
        if not isinstance(text, str):
            raise TypeError("selected member extract_text must return str")
        return scalar_text_result(text, source=member)

    @staticmethod
    def _selected_transformed_text(
        member: Any,
        request: Any,
        *,
        content_filter: ContentFilter | None,
    ) -> str | None:
        """Transform one literal contribution without flattening inner aggregates.

        An aggregate selected as one object still owns meaningful boundaries of
        its own.  Delegate the transform request to that aggregate so filtering,
        newline handling, and stripping continue to run at its leaves and never
        consume or rewrite its separators.
        """

        if isinstance(member, AggregateTextMixin):
            return member.extract_text(
                layout=False,
                apply_exclusions=True,
                newlines=request.newlines,
                whitespace=request.whitespace,
                strip=request.strip,
                bidi=False,
                content_filter=content_filter,
            )

        result = SelectedTextMixin._selected_raw_result(member)
        if result is None:
            return None
        return apply_prepared_text_transform(result.text, request)

    def extract_text(
        self,
        *,
        separator: str = " ",
        newlines: bool | str = True,
        whitespace: WhitespaceMode = "preserve",
        strip: bool = True,
        content_filter: ContentFilter | None = None,
    ) -> str:
        """Join selected textual contributions literally in stored order.

        Empty and non-text contributions are omitted. Duplicate selections are
        retained. Each contribution is transformed independently, so filters
        cannot match across element boundaries and separators are unchanged.
        """

        if not isinstance(separator, str):
            raise TypeError("separator must be a str")
        request = prepare_text_transform(
            newlines=newlines,
            whitespace=whitespace,
            strip=strip,
            bidi=False,
            content_filter=content_filter,
        )
        texts: list[str] = []
        for member in self._iter_selected_text_members():
            text = self._selected_transformed_text(
                member,
                request,
                content_filter=content_filter,
            )
            if text is None:
                continue
            if text:
                texts.append(text)
        return separator.join(texts)

    def extract_text_result(self, *, separator: str = " ") -> ExtractedText:
        """Join nonempty selected raw results with exact source offsets."""

        if not isinstance(separator, str):
            raise TypeError("separator must be a str")
        results: list[ExtractedText] = []
        for member in self._iter_selected_text_members():
            result = self._selected_raw_result(member)
            if result is not None and result.text:
                results.append(result)
        return join_extracted_text(results, separator=separator, preserve_empty=False)


__all__ = [
    "AggregateTextMixin",
    "ScalarTextMixin",
    "SelectedTextMixin",
    "SpatialTextMixin",
]
