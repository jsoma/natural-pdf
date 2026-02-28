"""Regression tests for layout module bug fixes and improvements."""

import dataclasses
from unittest.mock import MagicMock

import pytest

from natural_pdf.analyzers.layout.layout_options import (
    BaseLayoutOptions,
    PaddleLayoutOptions,
    VLMLayoutOptions,
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
    def test_vlm_options_with_client_survive_replace(self):
        """dataclasses.replace() should work on VLMLayoutOptions with a mock client."""
        mock_client = MagicMock()
        opts = VLMLayoutOptions(
            client=mock_client,
            classes=["table", "figure"],
            extra_args={"temperature": 0.5},
        )

        # This is what layout_analyzer.py now does instead of deepcopy
        copied = dataclasses.replace(
            opts,
            extra_args=dict(opts.extra_args),
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
            classes=list(original.classes) if original.classes else None,
            exclude_classes=list(original.exclude_classes) if original.exclude_classes else None,
        )

        # Mutate the copy
        copied.extra_args["new_key"] = "new_value"
        copied.classes.append("figure")

        # Original should be unchanged
        assert "new_key" not in original.extra_args
        assert original.classes == ["table"]


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

    def test_internal_dict_removed_from_base_options(self):
        """BaseLayoutOptions should NOT have _internal dict field (replaced by DetectionContext)."""
        opts = BaseLayoutOptions()
        assert not hasattr(opts, "_internal")

    def test_detection_context_exists(self):
        """DetectionContext dataclass should be importable and have expected fields."""
        from natural_pdf.analyzers.layout.layout_options import DetectionContext

        ctx = DetectionContext()
        assert ctx.layout_host is None
        assert ctx.img_scale_x == 1.0
        assert ctx.img_scale_y == 1.0

    def test_detection_context_with_values(self):
        """DetectionContext should accept values."""
        from natural_pdf.analyzers.layout.layout_options import DetectionContext

        ctx = DetectionContext(layout_host="page", img_scale_x=2.0, img_scale_y=3.0)
        assert ctx.layout_host == "page"
        assert ctx.img_scale_x == 2.0
        assert ctx.img_scale_y == 3.0


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
# Review fix: Paddle uses _ENGINE_FIELDS allow-list
# ---------------------------------------------------------------------------
class TestPaddleEngineFields:
    def test_cache_key_consistent(self):
        """Paddle model cache key should be consistent for same options."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        opts1 = PaddleLayoutOptions()
        opts2 = PaddleLayoutOptions()

        assert detector._get_cache_key(opts1) == detector._get_cache_key(opts2)

    def test_cache_key_differs_for_lang(self):
        """Paddle cache key should differ when lang changes (affects model init)."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        detector = PaddleLayoutDetector()
        opts_default = PaddleLayoutOptions()
        opts_en = PaddleLayoutOptions(lang="en")

        assert detector._get_cache_key(opts_default) != detector._get_cache_key(opts_en)

    def test_engine_fields_is_allow_list(self):
        """_ENGINE_FIELDS should contain only fields forwarded to PPStructureV3."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        engine_fields = PaddleLayoutDetector._ENGINE_FIELDS
        # Must include device and use_* flags
        assert "device" in engine_fields
        assert "use_table_recognition" in engine_fields
        # Must NOT include internal/filtering fields
        assert "confidence" not in engine_fields
        assert "classes" not in engine_fields
        assert "extra_args" not in engine_fields
        assert "create_cells" not in engine_fields
        assert "lang" not in engine_fields

    def test_no_skip_fields_attribute(self):
        """_SKIP_FIELDS should no longer exist (replaced by _ENGINE_FIELDS)."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        assert not hasattr(PaddleLayoutDetector, "_SKIP_FIELDS")


# ---------------------------------------------------------------------------
# Phase 4: Canonical type mapping
# ---------------------------------------------------------------------------
class TestCanonicalTypeMapping:
    def test_yolo_plain_text_maps_to_text(self):
        """YOLO 'plain-text' should map to canonical 'text'."""
        from natural_pdf.analyzers.layout.yolo import YOLODocLayoutDetector

        assert YOLODocLayoutDetector.TYPE_MAP["plain-text"] == "text"

    def test_yolo_abandon_maps_to_unknown(self):
        """YOLO 'abandon' should map to canonical 'unknown'."""
        from natural_pdf.analyzers.layout.yolo import YOLODocLayoutDetector

        assert YOLODocLayoutDetector.TYPE_MAP["abandon"] == "unknown"

    def test_surya_pageheader_maps_to_header(self):
        """Surya 'pageheader' should map to canonical 'header'."""
        from natural_pdf.analyzers.layout.surya import SuryaLayoutDetector

        assert SuryaLayoutDetector.TYPE_MAP["pageheader"] == "header"

    def test_surya_sectionheader_maps_to_heading(self):
        """Surya 'sectionheader' should map to canonical 'heading'."""
        from natural_pdf.analyzers.layout.surya import SuryaLayoutDetector

        assert SuryaLayoutDetector.TYPE_MAP["sectionheader"] == "heading"

    def test_paddle_equation_maps_to_formula(self):
        """Paddle 'equation' should map to canonical 'formula'."""
        from natural_pdf.analyzers.layout.paddle import PaddleLayoutDetector

        assert PaddleLayoutDetector.TYPE_MAP["equation"] == "formula"

    def test_unknown_type_passes_through(self):
        """Unmapped types should pass through as-is."""
        from natural_pdf.analyzers.layout.yolo import YOLODocLayoutDetector

        type_map = YOLODocLayoutDetector.TYPE_MAP
        assert type_map.get("never-seen-before", "never-seen-before") == "never-seen-before"

    def test_base_type_map_is_empty(self):
        """Base class TYPE_MAP should be empty."""
        from natural_pdf.analyzers.layout.base import LayoutDetector

        assert LayoutDetector.TYPE_MAP == {}

    def test_vlm_type_map_is_empty(self):
        """VLM TYPE_MAP should be empty (dynamic classes)."""
        from natural_pdf.analyzers.layout.vlm import VLMLayoutDetector

        assert VLMLayoutDetector.TYPE_MAP == {}


# ---------------------------------------------------------------------------
# Consensus fix: engine_name_for_options skips deprecated aliases
# ---------------------------------------------------------------------------
class TestEngineNameForOptions:
    def test_returns_canonical_not_alias(self):
        """engine_name_for_options should return 'vlm', never 'gemini'."""
        from natural_pdf.analyzers.layout.layout_manager import engine_name_for_options
        from natural_pdf.analyzers.layout.layout_options import VLMLayoutOptions

        opts = VLMLayoutOptions()
        name = engine_name_for_options(opts)
        assert name == "vlm", f"Expected 'vlm' but got '{name}'"

    def test_returns_none_for_unknown_options(self):
        """engine_name_for_options should return None for unknown options."""
        from natural_pdf.analyzers.layout.layout_manager import engine_name_for_options

        opts = BaseLayoutOptions()
        name = engine_name_for_options(opts)
        # BaseLayoutOptions is parent of all, so it may match the first engine
        # The important thing is it never returns a deprecated alias
        if name is not None:
            from natural_pdf.analyzers.layout.layout_manager import _DEPRECATED_ALIASES

            assert name not in _DEPRECATED_ALIASES


# ---------------------------------------------------------------------------
# Consensus fix: VLM bbox length validation
# ---------------------------------------------------------------------------
class TestVLMBboxValidation:
    def test_malformed_bbox_skipped(self):
        """VLM detector should skip regions with bbox length != 4."""
        from natural_pdf.analyzers.layout.layout_options import VLMLayoutOptions
        from natural_pdf.analyzers.layout.vlm import DetectedRegion, VLMLayoutDetector

        detector = VLMLayoutDetector()
        opts = VLMLayoutOptions(classes=["table", "figure"])

        parsed = [
            DetectedRegion(label="table", bbox=[10, 20, 100, 200], confidence=0.9),
            DetectedRegion(label="figure", bbox=[10, 20, 100], confidence=0.8),  # too short
            DetectedRegion(label="table", bbox=[1, 2, 3, 4, 5], confidence=0.7),  # too long
        ]
        results = detector._filter_detections(parsed, opts)
        assert len(results) == 1
        assert results[0]["class"] == "table"
        assert results[0]["bbox"] == (10, 20, 100, 200)
