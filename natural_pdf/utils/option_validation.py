# option_validation.py
"""
Validation helpers for Natural PDF options.

Provides warn-and-correct behavior for recoverable validation issues,
and raises InvalidOptionError for truly invalid states.

Design philosophy:
- Warn + auto-correct for recoverable issues (out-of-range values)
- Raise errors only for truly invalid states (non-existent paths)
- Standard warning format: [ClassName] field_name=value reason, using corrected_value
- Strict mode via NATURAL_PDF_STRICT=1 environment variable
"""

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Valid device options
VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}


def is_strict_mode() -> bool:
    """Check if strict validation mode is enabled via environment variable."""
    return os.environ.get("NATURAL_PDF_STRICT", "").lower() in ("1", "true", "yes")


def _format_warning(
    class_name: str, field_name: str, original: Any, reason: str, corrected: Any
) -> str:
    """Format a standardized warning message."""
    return f"[{class_name}] {field_name}={original!r} {reason}, using {corrected!r}"


def _handle_validation(
    class_name: str,
    field_name: str,
    original: Any,
    reason: str,
    corrected: Any,
) -> Any:
    """Handle a validation issue: warn and return corrected value, or raise in strict mode."""
    from natural_pdf.exceptions import InvalidOptionError

    message = _format_warning(class_name, field_name, original, reason, corrected)

    if is_strict_mode():
        raise InvalidOptionError(message)

    logger.warning(message)
    return corrected


def validate_confidence(
    value: float,
    field_name: str = "confidence",
    class_name: str = "Options",
) -> float:
    """
    Validate confidence is 0.0-1.0, warn and clamp if not.

    Args:
        value: The confidence value to validate
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages

    Returns:
        Validated confidence value (clamped to 0.0-1.0 if needed)
    """
    if value is None:
        return value

    # Try to coerce to float if needed
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
            logger.warning(
                _format_warning(class_name, field_name, value, "coerced to float", value)
            )
        except (ValueError, TypeError):
            return _handle_validation(class_name, field_name, value, "cannot convert to float", 0.5)

    if value < 0.0:
        return _handle_validation(class_name, field_name, value, "< 0.0", 0.0)

    if value > 1.0:
        return _handle_validation(class_name, field_name, value, "> 1.0", 1.0)

    return float(value)


def validate_positive_int(
    value: int,
    field_name: str,
    class_name: str = "Options",
    default: int = 1,
) -> int:
    """
    Validate value > 0, warn and use default if not.

    Args:
        value: The integer value to validate
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages
        default: Default value to use if validation fails

    Returns:
        Validated positive integer
    """
    if value is None:
        return value

    # Try to coerce to int if needed
    if not isinstance(value, int) or isinstance(value, bool):
        try:
            original = value
            value = int(value)
            logger.warning(
                _format_warning(class_name, field_name, original, "coerced to int", value)
            )
        except (ValueError, TypeError):
            return _handle_validation(
                class_name, field_name, value, "cannot convert to int", default
            )

    if value <= 0:
        return _handle_validation(class_name, field_name, value, "<= 0", default)

    return value


def validate_device(
    value: str,
    field_name: str = "device",
    class_name: str = "Options",
) -> str:
    """
    Validate device is cpu/cuda/mps/auto, warn and use 'cpu' if not.

    Args:
        value: The device string to validate
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages

    Returns:
        Validated device string
    """
    if value is None:
        return value

    if not isinstance(value, str):
        return _handle_validation(class_name, field_name, value, "must be string", "cpu")

    value_lower = value.lower().strip()

    # Accept cuda:N format
    if value_lower.startswith("cuda:"):
        return value

    if value_lower not in VALID_DEVICES:
        return _handle_validation(class_name, field_name, value, f"not in {VALID_DEVICES}", "cpu")

    return value


def validate_non_empty_string(
    value: str,
    field_name: str,
    class_name: str = "Options",
    default: str = "",
) -> str:
    """
    Validate value is a non-empty string.

    Args:
        value: The string value to validate
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages
        default: Default value to use if validation fails

    Returns:
        Validated non-empty string
    """
    if value is None:
        return value

    if not isinstance(value, str):
        return _handle_validation(class_name, field_name, value, "must be string", default)

    if not value.strip():
        return _handle_validation(class_name, field_name, value, "is empty", default)

    return value


def validate_path_exists(
    value: Union[str, Path],
    field_name: str,
    class_name: str = "Options",
) -> Union[str, Path]:
    """
    Validate that a path exists. Raises InvalidOptionError if not.

    This is a hard error - there's no safe default for a path that should exist.

    Args:
        value: The path to validate
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages

    Returns:
        The validated path

    Raises:
        InvalidOptionError: If the path does not exist
    """
    from natural_pdf.exceptions import InvalidOptionError

    if value is None:
        return value

    path = Path(value)
    if not path.exists():
        raise InvalidOptionError(f"[{class_name}] {field_name}={value!r} does not exist")

    return value


def coerce_to_float(
    value: Any,
    field_name: str,
    class_name: str = "Options",
    default: float = 0.0,
) -> float:
    """
    Coerce value to float, warn if coerced.

    Args:
        value: The value to coerce
        field_name: Name of the field for error messages
        class_name: Name of the class for error messages
        default: Default value if coercion fails

    Returns:
        Float value
    """
    if value is None:
        return value

    if isinstance(value, float):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    try:
        result = float(value)
        logger.warning(_format_warning(class_name, field_name, value, "coerced to float", result))
        return result
    except (ValueError, TypeError):
        return _handle_validation(class_name, field_name, value, "cannot convert to float", default)


def validate_option_type(
    options: Any,
    expected_type: type,
    engine_name: str,
) -> Tuple[Any, bool]:
    """
    Validate options type and log warning if wrong type passed.

    Args:
        options: The options object to validate
        expected_type: The expected type
        engine_name: Name of the engine for error messages

    Returns:
        Tuple of (options, was_default_used) - if wrong type, returns
        a new instance of expected_type and True
    """
    if isinstance(options, expected_type):
        return options, False

    logger.warning(
        f"[{engine_name}] Expected {expected_type.__name__}, got {type(options).__name__}. "
        f"Using default {expected_type.__name__}."
    )
    return expected_type(), True
