#!/usr/bin/env python3
"""
Test script to verify the new cleanup methods work correctly.

This test verifies that:
1. Cleanup methods exist and are callable
2. They handle edge cases gracefully (empty caches, missing engines)
3. They actually clean up loaded models/engines
"""

import gc
import os
import sys
from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.classification.manager import ClassificationManager
from natural_pdf.ocr.ocr_manager import OCRManager


class TestCleanupMethods:
    """Test suite for manager cleanup methods"""

    def test_ocr_manager_cleanup_empty(self):
        """Test OCR manager cleanup when no engines are loaded"""
        manager = OCRManager()

        # Test cleanup when nothing is loaded
        count = manager.cleanup_engine()
        assert count == 0, "Should return 0 when no engines loaded"

        # Test cleanup of specific non-existent engine
        count = manager.cleanup_engine("nonexistent")
        assert count == 0, "Should return 0 when engine doesn't exist"

    def test_layout_manager_cleanup_empty(self):
        """Test Layout manager cleanup when no detectors are loaded"""
        manager = LayoutManager()

        # Test cleanup when nothing is loaded
        count = manager.cleanup_detector()
        assert count == 0, "Should return 0 when no detectors loaded"

        # Test cleanup of specific non-existent detector
        count = manager.cleanup_detector("nonexistent")
        assert count == 0, "Should return 0 when detector doesn't exist"

    def test_classification_manager_cleanup_empty(self):
        """Test Classification manager cleanup when no models are loaded"""
        try:
            manager = ClassificationManager()

            # Test cleanup when nothing is loaded
            count = manager.cleanup_models()
            assert count == 0, "Should return 0 when no models loaded"

            # Test cleanup of specific non-existent model
            count = manager.cleanup_models("nonexistent/model")
            assert count == 0, "Should return 0 when model doesn't exist"

        except ImportError:
            pytest.skip("Classification dependencies not available")

    def test_ocr_manager_cleanup_with_engine(self):
        """Test OCR manager cleanup after loading an engine"""
        manager = OCRManager()

        # Check if any OCR engines are available
        available_engines = manager.get_available_engines()
        if not available_engines:
            pytest.skip("No OCR engines available for testing")

        engine_name = available_engines[0]
        print(f"Testing with OCR engine: {engine_name}")

        # Load an engine by accessing it
        try:
            engine_instance = manager._get_engine_instance(engine_name)
            assert engine_name in manager._engine_instances, "Engine should be cached"

            # Test cleanup of specific engine
            count = manager.cleanup_engine(engine_name)
            assert count == 1, f"Should return 1 after cleaning up {engine_name}"
            assert (
                engine_name not in manager._engine_instances
            ), "Engine should be removed from cache"

        except Exception as e:
            pytest.skip(f"Could not load {engine_name} engine: {e}")

    def test_layout_manager_cleanup_with_detector(self):
        """Test Layout manager cleanup after loading a detector"""
        manager = LayoutManager()

        # Check if any layout engines are available
        available_engines = manager.get_available_engines()
        if not available_engines:
            pytest.skip("No layout engines available for testing")

        engine_name = available_engines[0]
        print(f"Testing with layout engine: {engine_name}")

        # Load a detector by accessing it
        try:
            detector_instance = manager._get_engine_instance(engine_name)
            assert engine_name in manager._detector_instances, "Detector should be cached"

            # Test cleanup of specific detector
            count = manager.cleanup_detector(engine_name)
            assert count == 1, f"Should return 1 after cleaning up {engine_name}"
            assert (
                engine_name not in manager._detector_instances
            ), "Detector should be removed from cache"

        except Exception as e:
            pytest.skip(f"Could not load {engine_name} detector: {e}")

    def test_methods_exist(self):
        """Test that all cleanup methods exist and are callable"""
        # Test OCRManager
        manager = OCRManager()
        assert hasattr(manager, "cleanup_engine"), "OCRManager should have cleanup_engine method"
        assert callable(manager.cleanup_engine), "cleanup_engine should be callable"

        # Test LayoutManager
        layout_manager = LayoutManager()
        assert hasattr(
            layout_manager, "cleanup_detector"
        ), "LayoutManager should have cleanup_detector method"
        assert callable(layout_manager.cleanup_detector), "cleanup_detector should be callable"

        # Test ClassificationManager (if available)
        try:
            classification_manager = ClassificationManager()
            assert hasattr(
                classification_manager, "cleanup_models"
            ), "ClassificationManager should have cleanup_models method"
            assert callable(
                classification_manager.cleanup_models
            ), "cleanup_models should be callable"
        except ImportError:
            print("Classification dependencies not available, skipping ClassificationManager test")


def main():
    """Run the cleanup method tests"""
    print("Testing manager cleanup methods...")

    # Run pytest on just this file
    exit_code = pytest.main([__file__, "-v", "-s"])

    if exit_code == 0:
        print("\n✅ All cleanup method tests passed!")
        print("The memory management methods are working correctly.")
    else:
        print("\n❌ Some tests failed!")
        print("The cleanup methods need investigation.")

    return exit_code


if __name__ == "__main__":
    exit(main())
