"""
Utilities for generating consistent identifiers.
"""

import base64
import hashlib
import os


def generate_short_path_hash(path_str: str, length: int = 8) -> str:
    """
    Generates a short, filesystem-safe hash ID from a path string.

    Args:
        path_str: The absolute path string.
        length: The desired length of the short ID (default: 8).

    Returns:
        A short hash string using URL-safe Base64 encoding.
    """
    # Ensure consistency by using the absolute path
    normalized_path = os.path.abspath(path_str)
    path_bytes = normalized_path.encode("utf-8")
    # Use SHA-256 for good collision resistance
    full_hash = hashlib.sha256(path_bytes).digest()  # Get binary hash
    # Encode using URL-safe Base64 and remove padding '=' characters
    b64_encoded = base64.urlsafe_b64encode(full_hash).decode("ascii").rstrip("=")
    # Return the first 'length' characters
    if length <= 0 or length > len(b64_encoded):
        raise ValueError(
            f"Invalid length specified: {length}. Must be between 1 and {len(b64_encoded)}."
        )
    return b64_encoded[:length]
