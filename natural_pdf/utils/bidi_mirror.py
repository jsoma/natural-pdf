"""Light-weight bracket mirroring for RTL text.

This module provides `mirror_brackets`, a fast pure-python helper that
replaces each bracket/parenthesis character with its Unicode-defined pair.

For everyday PDFs the six ASCII pairs are enough, but the mapping can be
extended easily from Unicode's BidiBrackets.txt.
"""

from typing import Dict

# Minimal mapping – ( ) [ ] { }
_ASCII_MIRROR: Dict[int, str] = {
    0x0028: ")",  # ( -> )
    0x0029: "(",  # ) -> (
    0x005B: "]",  # [ -> ]
    0x005D: "[",  # ] -> [
    0x007B: "}",  # { -> }
    0x007D: "{",  # } -> {
}


def mirror_brackets(text: str) -> str:  # pragma: no cover
    """Return *text* with each bracket replaced by its mirror partner.

    The function is context-free: it blindly flips every character found in
    the mapping, which is sufficient once the string is already in visual
    order (e.g., after `bidi.algorithm.get_display`).
    """
    if not text:
        return text
    # Fast path: only allocate when needed
    out_chars = []
    append = out_chars.append
    for ch in text:
        append(_ASCII_MIRROR.get(ord(ch), ch))
    return "".join(out_chars)
