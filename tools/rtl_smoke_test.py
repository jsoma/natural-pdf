#!/usr/bin/env python3
"""RTL pipeline smoke-test for natural-pdf.

Run it from the repository root:

    python tools/rtl_smoke_test.py

It loads *pdfs/arabic.pdf* and performs a handful of checks that cover the
most common break-points we identified for RTL handling:
    1. char ingestion / word grouping
    2. selector finds on logical Arabic tokens
    3. bracket mirroring
    4. number directionality inside RTL lines

Exit code is **0** when all checks pass, **1** otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

from bidi.algorithm import get_display  # type: ignore

from natural_pdf import PDF
from natural_pdf.utils.bidi_mirror import mirror_brackets


PDF_PATH = Path("pdfs/arabic.pdf")

if not PDF_PATH.exists():
    print(f"❗  PDF not found: {PDF_PATH.resolve()}")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

failures: list[str] = []

def check(cond: bool, msg: str):
    """Collect failures but keep running to show full report."""
    if cond:
        print(f"✓ {msg}")
    else:
        print(f"✗ {msg}")
        failures.append(msg)


# ────────────────────────────────────────────────────────────────
# Load page
# ────────────────────────────────────────────────────────────────

pdf = PDF(str(PDF_PATH))
page = pdf.pages[0]

# Basic char/word counts (should be non-zero)
check(len(page.chars) > 0, "chars were ingested")
check(len(page.words) > 0, "words were grouped")

# First line logical text
logical_first_line = page.extract_text().split("\n")[0]
print("First logical line:")
print("  ", logical_first_line)

# 1. Arabic keyword should be findable
check(page.find(text="مكرر") is not None, "page.find works for Arabic token 'مكرر'")

# 2. Reversed token should NOT match
check(page.find(text="مكرر"[::-1]) is None, "reverse token does not match (logical order stored)")

# 3. Extracted line should already show the bracket pair in correct orientation
check("(مكرر)" in logical_first_line, "parentheses orientation is correct in extract_text")

# 4. Western numbers must stay LTR inside RTL
#    After visual re-order, the line should end with 2022 (year on the left visually → last in logical string)
check(logical_first_line.rstrip().endswith("2022"), "Western number '2022' kept logical placement")

print("\nSummary: {} passed, {} failed".format(4 - len(failures), len(failures)))

sys.exit(0 if not failures else 1) 