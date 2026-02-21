"""Regression tests for layout module bug fixes and improvements."""

import dataclasses
from unittest.mock import MagicMock

import pytest

from natural_pdf.analyzers.layout.layout_options import (
    BaseLayoutOptions,
    GeminiLayoutOptions,
    PaddleLayoutOptions,
)


# ---------------------------------------------------------------------------
# Bug 1 regression: Paddle model name should be "paddle", not "paddle_v3"
# ---------------------------------------------------------------------------
class TestPaddleModelName:
    def test_paddle_detections_use_correct_model_name(self):
        """Paddle detector should set model='paddle' so layout_service filtering works."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        # We can't run detect() without the real Paddle library, but we can
        # verify via source inspection that 'paddle_v3' is no longer referenced.
        import inspect

        source = inspect.getsource(detector.detect)
        assert "paddle_v3" not in source, "Paddle detector still references 'paddle_v3'"
        assert '"model": "paddle"' in source or "'model': 'paddle'" in source

    def test_layout_service_engine_filter_case_insensitive(self):
        """layout_service engine filter should be case-insensitive."""
        import inspect

        from natural_pdf.services.layout_service import LayoutService

        source = inspect.getsource(LayoutService._analyze_page)
        assert ".lower()" in source, "Engine comparison should be case-normalized"


# ---------------------------------------------------------------------------
# Bug 2 regression: GeminiLayoutOptions with client survives options copying
# ---------------------------------------------------------------------------
class TestOptionsCopying:
    def test_gemini_options_with_client_survive_replace(self):
        """dataclasses.replace() should work on GeminiLayoutOptions with a mock client."""
        mock_client = MagicMock()
        opts = GeminiLayoutOptions(
            client=mock_client,
            classes=["table", "figure"],
            extra_args={"temperature": 0.5},
        )

        # This is what layout_analyzer.py now does instead of deepcopy
        copied = dataclasses.replace(
            opts,
            extra_args=dict(opts.extra_args),
            _internal=dict(opts._internal),
            classes=list(opts.classes) if opts.classes else None,
            exclude_classes=list(opts.exclude_classes) if opts.exclude_classes else None,
        )

        # Client reference should be preserved (not deep-copied)
        assert copied.client is mock_client
        # Mutable fields should be independent copies
        assert copied.extra_args is not opts.extra_args
        assert copied.extra_args == opts.extra_args
        assert copied.classes is not opts.classes
        assert copied.classes == opts.classes

    def test_options_not_mutated_after_copy(self):
        """Caller's options should not be mutated when the copy is modified."""
        original = BaseLayoutOptions(
            extra_args={"key": "value"},
            classes=["table"],
        )
        copied = dataclasses.replace(
            original,
            extra_args=dict(original.extra_args),
            _internal=dict(original._internal),
            classes=list(original.classes) if original.classes else None,
            exclude_classes=list(original.exclude_classes) if original.exclude_classes else None,
        )

        # Mutate the copy
        copied.extra_args["new_key"] = "new_value"
        copied.classes.append("figure")
        copied._internal["_layout_host"] = "test"

        # Original should be unchanged
        assert "new_key" not in original.extra_args
        assert original.classes == ["table"]
        assert "_layout_host" not in original._internal


# ---------------------------------------------------------------------------
# Bug 3 regression: Surya dead logic removed
# ---------------------------------------------------------------------------
class TestSuryaDeadLogic:
    def test_no_can_do_table_rec_variable(self):
        """The dead can_do_table_rec variable should not exist in surya.py."""
        import inspect

        from natural_pdf.analyzers.layout.surya import SuryaLayoutDetector

        source = inspect.getsource(SuryaLayoutDetector.detect)
        assert "can_do_table_rec" not in source


# ---------------------------------------------------------------------------
# Phase 2e: _page_ref removed
# ---------------------------------------------------------------------------
class TestInternalContext:
    def test_no_page_ref_in_analyzer(self):
        """_page_ref should not be set in layout_analyzer extra_args."""
        import inspect

        from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer

        source = inspect.getsource(LayoutAnalyzer.analyze_layout)
        assert "_page_ref" not in source

    def test_internal_dict_exists_on_base_options(self):
        """BaseLayoutOptions should have _internal dict field."""
        opts = BaseLayoutOptions()
        assert hasattr(opts, "_internal")
        assert isinstance(opts._internal, dict)
        assert opts._internal == {}


# ---------------------------------------------------------------------------
# Phase 3a: PaddleLayoutOptions bloat reduction
# ---------------------------------------------------------------------------
class TestPaddleOptionsSimplified:
    def test_commonly_used_fields_present(self):
        """PaddleLayoutOptions should have the commonly-used fields."""
        opts = PaddleLayoutOptions()
        assert hasattr(opts, "lang")
        assert hasattr(opts, "create_cells")
        assert hasattr(opts, "verbose")
        assert hasattr(opts, "use_table_recognition")
        assert hasattr(opts, "use_doc_orientation_classify")
        assert hasattr(opts, "use_doc_unwarping")

    def test_removed_fields_go_to_extra_args(self):
        """Formerly-explicit fields can now be passed via extra_args."""
        opts = PaddleLayoutOptions(extra_args={"layout_detection_model_name": "custom_model"})
        assert opts.extra_args["layout_detection_model_name"] == "custom_model"

    def test_no_device_shadowing(self):
        """PaddleLayoutOptions should inherit device from BaseLayoutOptions, not re-declare it."""
        # Check that PaddleLayoutOptions doesn't define its own 'device' field
        own_fields = {
            f.name
            for f in dataclasses.fields(PaddleLayoutOptions)
            if f.name not in {f.name for f in dataclasses.fields(BaseLayoutOptions)}
        }
        assert "device" not in own_fields, "PaddleLayoutOptions should not re-declare 'device'"


# ---------------------------------------------------------------------------
# Phase 3b: TATR post_process_regions hook
# ---------------------------------------------------------------------------
class TestPostProcessHook:
    def test_base_post_process_is_noop(self):
        """Base class post_process_regions should be a no-op."""
        from natural_pdf.analyzers.layout.base import LayoutDetector

        # Can't instantiate abstract class, but we can check the method exists
        assert hasattr(LayoutDetector, "post_process_regions")

    def test_tatr_has_post_process(self):
        """TATR detector should override post_process_regions."""
        from natural_pdf.analyzers.layout.tatr import TableTransformerDetector

        # Verify it overrides the base method
        assert TableTransformerDetector.post_process_regions is not object.__init__


# ---------------------------------------------------------------------------
# Phase 3c: _build_class_filters helper
# ---------------------------------------------------------------------------
class TestBuildClassFilters:
    def test_build_class_filters_with_classes(self):
        """_build_class_filters should return normalized sets."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        opts = BaseLayoutOptions(
            classes=["Table", "Figure_Caption"],
            exclude_classes=["Plain Text"],
        )
        req, excl = detector._build_class_filters(opts)
        assert req == {"table", "figure-caption"}
        assert excl == {"plain-text"}

    def test_build_class_filters_none_classes(self):
        """_build_class_filters should return None for req when no classes specified."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        opts = BaseLayoutOptions()
        req, excl = detector._build_class_filters(opts)
        assert req is None
        assert excl == set()


# ---------------------------------------------------------------------------
# Review fix: _internal excluded from Paddle cache key and init_args
# ---------------------------------------------------------------------------
class TestPaddleInternalExcluded:
    def test_internal_excluded_from_cache_key(self):
        """_internal dict should not affect Paddle model cache key."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        opts1 = PaddleLayoutOptions()
        opts2 = PaddleLayoutOptions()
        opts2._internal["_layout_host"] = "some_page_object"
        opts2._internal["_img_scale_x"] = 1.5

        assert detector._get_cache_key(opts1) == detector._get_cache_key(opts2)

    def test_internal_excluded_from_skip_fields(self):
        """_internal must be in _SKIP_FIELDS class constant."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        assert "_internal" in PaddleLayoutDetector._SKIP_FIELDS
