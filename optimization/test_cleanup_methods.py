#!/usr/bin/env python3
"""
Test script to verify the new cleanup methods work correctly.

This test verifies that:
1. Cleanup methods exist and are callable
2. They handle edge cases gracefully (empty caches, missing engines)
3. They actually clean up loaded models/engines
"""


import threading

import pytest

import natural_pdf.ocr.ocr_manager as ocr_registry
import natural_pdf.ocr.ocr_provider as ocr_provider
from natural_pdf.classification.pipelines import cleanup_models, is_classification_available
from natural_pdf.ocr.ocr_manager import cleanup_engine


class TestCleanupMethods:
    """Test suite for manager cleanup methods"""

    def test_ocr_cleanup_empty(self, monkeypatch):
        """Test OCR cleanup helpers when no engines are loaded"""
        monkeypatch.setattr(ocr_provider, "_engine_instances", {})
        monkeypatch.setattr(ocr_provider, "_engine_locks", {})
        monkeypatch.setattr(ocr_provider, "_engine_inference_locks", {})

        assert cleanup_engine() == 0, "Should return 0 when no engines loaded"
        assert cleanup_engine("nonexistent") == 0, "Should return 0 when engine doesn't exist"

    def test_classification_cleanup_empty(self):
        """Test classification cleanup when no models are loaded"""
        if not is_classification_available():
            pytest.skip("Classification dependencies not available")

        # Cleanup when nothing cached
        count = cleanup_models()
        assert count == 0, "Should return 0 when no models loaded"

        count = cleanup_models("nonexistent/model")
        assert count == 0, "Should return 0 when model doesn't exist"

    def test_ocr_cleanup_with_engine(self, monkeypatch):
        """Test OCR cleanup after manually caching an engine"""

        class _DummyEngine:
            def __init__(self):
                self.cleaned = False

            def cleanup(self):
                self.cleaned = True

        instances = {}
        locks = {}
        inference_locks = {}
        monkeypatch.setattr(ocr_provider, "_engine_instances", instances)
        monkeypatch.setattr(ocr_provider, "_engine_locks", locks)
        monkeypatch.setattr(ocr_provider, "_engine_inference_locks", inference_locks)

        dummy_name = "__test_engine__"
        dummy_engine = _DummyEngine()
        instances[dummy_name] = dummy_engine
        locks[dummy_name] = threading.Lock()
        inference_locks[dummy_name] = threading.Lock()

        count = cleanup_engine(dummy_name)
        assert count == 1, "Should report one cleaned engine"
        assert dummy_engine.cleaned, "Cleanup hook should be invoked"
        assert dummy_name not in instances

    def test_methods_exist(self):
        """Test that all cleanup methods exist and are callable"""
        # Test OCR cleanup helper existence
        assert callable(cleanup_engine), "cleanup_engine helper should be callable"

        assert callable(cleanup_models), "cleanup_models helper should be callable"


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
