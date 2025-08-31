"""Monkey patches for pdfminer.six bugs.

This module contains patches for known bugs in pdfminer.six that affect
natural_pdf functionality. These patches are applied automatically when
natural_pdf is imported.
"""

import logging
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False

# Allow disabling patches via environment variable
import os

DISABLE_PATCHES = os.environ.get("NATURAL_PDF_DISABLE_PDFMINER_PATCHES", "").lower() in (
    "1",
    "true",
    "yes",
)


def _patch_color_space_bug():
    """
    Fix pdfminer.six color parsing bug for bare 'sc' commands.

    Bug: When a PDF uses 'sc' without an explicit color space (e.g., '1 1 0 sc'),
    pdfminer defaults to DeviceGray (1 component) and only reads one value,
    resulting in wrong colors.

    This patch detects when there are more color components on the stack than
    expected and handles RGB colors correctly.

    Reference: https://github.com/jsvine/pdfplumber/issues/XXX
    """
    try:
        import pdfminer.pdfinterp
        from pdfminer.casting import safe_rgb

        # Save original method
        original_do_scn = pdfminer.pdfinterp.PDFPageInterpreter.do_scn

        def patched_do_scn(self):
            """Patched do_scn that handles RGB colors without explicit color space."""
            # Get expected components from current color space
            n = self.graphicstate.ncs.ncomponents

            # Special handling for DeviceGray with potential RGB values
            if n == 1 and len(self.argstack) >= 3:
                # Peek at the last 3 values
                last_three = self.argstack[-3:]

                # Check if they look like RGB values (all numeric, 0-1 range)
                try:
                    values = []
                    for v in last_three:
                        if isinstance(v, (int, float)):
                            values.append(float(v))
                        else:
                            # Not numeric, use original behavior
                            return original_do_scn(self)

                    # If all values are in 0-1 range, treat as RGB
                    if all(0 <= v <= 1 for v in values):
                        # Pop 3 values and set as RGB
                        components = self.pop(3)
                        rgb = safe_rgb(*components)
                        if rgb is not None:
                            self.graphicstate.ncolor = rgb
                            return

                except (ValueError, TypeError, AttributeError):
                    # Any error, fall back to original
                    pass

            # Use original behavior for all other cases
            return original_do_scn(self)

        # Apply the patch
        pdfminer.pdfinterp.PDFPageInterpreter.do_scn = patched_do_scn
        logger.debug("Applied pdfminer color space bug patch")
        return True

    except Exception as e:
        logger.warning(f"Failed to apply pdfminer color patch: {e}")
        return False


def apply_patches():
    """Apply all pdfminer patches. Safe to call multiple times."""
    global _patches_applied

    if _patches_applied or DISABLE_PATCHES:
        return

    patches = [
        ("color_space_bug", _patch_color_space_bug),
        # Add more patches here as needed
    ]

    applied = []
    failed = []

    for name, patch_func in patches:
        if patch_func():
            applied.append(name)
        else:
            failed.append(name)

    if applied:
        logger.info(f"Applied pdfminer patches: {', '.join(applied)}")
    if failed:
        logger.warning(f"Failed to apply patches: {', '.join(failed)}")

    _patches_applied = True


def get_patch_status() -> dict:
    """Get information about applied patches."""
    return {
        "patches_applied": _patches_applied,
        "pdfminer_version": _get_pdfminer_version(),
    }


def _get_pdfminer_version() -> str:
    """Get the installed pdfminer version."""
    try:
        import pdfminer

        return getattr(pdfminer, "__version__", "unknown")
    except ImportError:
        return "not installed"
