"""Text analysis utilities for detecting VLM hallucination and repetition."""

import logging
import zlib

logger = logging.getLogger(__name__)


def detect_repetition(
    text: str,
    min_length: int = 500,
    threshold: float = 0.10,
) -> bool:
    """Detect repetitive/hallucinated text using compression ratio.

    Repetitive text (e.g., VLM hallucination loops) compresses extremely well
    because zlib exploits repeated byte patterns. Natural language typically
    compresses to a ratio of 0.3-0.6, structured HTML tables to ~0.12,
    while repetitive garbage compresses below 0.02.

    Args:
        text: The text to check.
        min_length: Minimum text length to bother checking. Short text
            can have low ratios naturally.
        threshold: Compression ratio below which text is considered
            repetitive. Default 0.10 catches clear repetition loops
            while passing structured content like HTML tables.

    Returns:
        True if the text appears to be repetitive/hallucinated.
    """
    if len(text) < min_length:
        return False

    compressed_size = len(zlib.compress(text.encode("utf-8")))
    ratio = compressed_size / len(text)
    logger.debug("Compression ratio for %d-char text: %.3f", len(text), ratio)
    return ratio < threshold
