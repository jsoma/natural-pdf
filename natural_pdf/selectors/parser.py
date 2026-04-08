"""CSS-like selector parser for natural-pdf.

This module implements a sophisticated selector parsing system that enables
jQuery-style element selection in PDF documents. It supports complex CSS-like
selectors with extensions for PDF-specific attributes and spatial relationships.

The parser handles:
- Basic element selectors (text, rect, line, image)
- Attribute selectors with comparisons ([size>12], [color="red"])
- Pseudo-selectors for text content (:contains(), :regex(), :closest())
- Spatial relationship selectors (:above(), :below(), :near())
- Color matching with Delta E distance calculations
- Logical operators (AND, OR) and grouping
- Complex nested expressions with proper precedence
- Fuzzy text matching for OCR errors (:closest())

Key features:
- Safe value parsing without eval() for security
- Color parsing from multiple formats (hex, RGB, names, CSS functions)
- Font and style attribute matching
- Coordinate and dimension-based selections
- Performance-optimized filtering functions

This enables powerful document navigation like:
- page.find('text[size>12]:bold:contains("Summary")')
- page.find_all('rect[color~="red"]:above(text:contains("Total"))')
- page.find('text:regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")')
- page.find('text:regex("[\u2500-\u257f]")')  # Box drawing characters
- page.find('text:closest("Date(s) of Review")')  # Fuzzy match for OCR errors
- page.find('text:closest("Invoice Date@0.9")')   # 90% similarity threshold
"""

import ast
import functools
import logging
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

from natural_pdf.selectors.registry import (
    ClauseEvalContext,
    get_attribute_handler,
    get_post_handler,
    get_pseudo_handler,
    get_relational_handler,
)

# colormath2 and colour are lazy-imported in color utility functions below
# to avoid pulling in networkx (~94ms) at startup.


logger = logging.getLogger(__name__)


def _normalize_selector_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def build_text_contains_selector(text_input: Union[str, Iterable[str]]) -> str:
    """
    Build a selector string that matches one or more literal text values.

    Args:
        text_input: A single string or an iterable of strings to match.

    Returns:
        Selector string using OR logic to match any of the provided texts.

    Raises:
        TypeError: If any value is not a string or text_input is not iterable.
        ValueError: If an iterable is provided but is empty.
    """
    if isinstance(text_input, str):
        texts = [text_input]
    else:
        if not isinstance(text_input, Iterable):
            raise TypeError("text must be a string or an iterable of strings.")
        texts = list(text_input)
        if not texts:
            raise ValueError("text iterable must contain at least one string.")
        for idx, value in enumerate(texts):
            if not isinstance(value, str):
                raise TypeError(
                    f"text iterable entries must be strings. "
                    f"Received {type(value).__name__} at index {idx}."
                )

    selectors: List[str] = []
    for value in texts:
        escaped_text = value.replace('"', '\\"').replace("'", "\\'")
        selectors.append(f'text:contains("{escaped_text}")')
    return "|".join(selectors)


def safe_parse_value(value_str: str) -> Any:
    """Safely parse a value string without using eval().

    Parses various value formats commonly found in PDF attributes while maintaining
    security by avoiding eval(). Supports numbers, tuples, lists, booleans, and
    quoted strings with proper type conversion.

    Args:
        value_str: String representation of a value. Can be a number ("12"),
            tuple ("(1.0, 0.5, 0.2)"), list ("[1, 2, 3]"), quoted string
            ('"Arial"'), boolean ("True"), or plain string ("Arial").

    Returns:
        Parsed value with appropriate Python type. Numbers become int/float,
        tuples/lists maintain structure, quoted strings are unquoted, and
        unrecognized values are returned as strings.

    Example:
        ```python
        safe_parse_value("12")          # -> 12
        safe_parse_value("12.5")        # -> 12.5
        safe_parse_value("(1,0,0)")     # -> (1, 0, 0)
        safe_parse_value('"Arial"')     # -> "Arial"
        safe_parse_value("True")        # -> True
        safe_parse_value("plain_text")  # -> "plain_text"
        ```

    Note:
        This function deliberately avoids eval() for security reasons and uses
        ast.literal_eval() for safe parsing of Python literals.
    """
    # Strip quotes first if it's a quoted string
    value_str = value_str.strip()
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Try parsing as a Python literal (numbers, tuples, lists)
    try:
        return ast.literal_eval(value_str)
    except (SyntaxError, ValueError):
        # If it's not a valid Python literal, return as is
        return value_str


def _parse_aggregate_function(value_str: str) -> Optional[Dict[str, Any]]:
    """Parse aggregate function syntax like min(), max(), avg(), closest("red").

    Also supports arithmetic expressions like:
    - max()*0.9  (90% of maximum)
    - avg()+10   (average plus 10)
    - min()-5    (minimum minus 5)

    Returns:
        Dict with 'type': 'aggregate', 'func': function name, 'args': optional args,
        and optionally 'operator' and 'operand' for arithmetic expressions.
        Returns None if not an aggregate function.
    """
    value_str = value_str.strip()

    # First try to match aggregate function with optional arithmetic expression
    # Pattern: funcname() or funcname(args) optionally followed by operator and number
    full_pattern = re.match(
        r"^(min|max|avg|mean|median|mode|most_common|closest)\s*\((.*?)\)\s*([\+\-\*/])?\s*([0-9.]+)?$",
        value_str,
        re.IGNORECASE,
    )

    if not full_pattern:
        return None

    func_name = full_pattern.group(1).lower()
    args_str = full_pattern.group(2).strip()
    operator = full_pattern.group(3)  # May be None
    operand_str = full_pattern.group(4)  # May be None

    # Normalize function aliases
    if func_name == "mean":
        func_name = "avg"
    elif func_name == "most_common":
        func_name = "mode"

    # Parse arguments if present
    args = None
    if args_str:
        # For closest(), parse the color argument
        if func_name == "closest":
            args = safe_parse_color(args_str)
        else:
            args = safe_parse_value(args_str)

    result = {"type": "aggregate", "func": func_name, "args": args}

    # Add arithmetic operation if present
    if operator and operand_str:
        try:
            operand = float(operand_str)
            result["operator"] = operator
            result["operand"] = operand
        except ValueError:
            # If we can't parse the operand as a number, ignore the arithmetic part
            pass

    return result


def safe_parse_color(value_str: str) -> tuple[float, float, float]:
    """
    Parse a color value which could be an RGB tuple, color name, hex code, or CSS-style rgb(...)/rgba(...).

    Args:
        value_str: String representation of a color (e.g., "red", "#ff0000", "(1,0,0)", "rgb(0,0,255)")

    Returns:
        RGB tuple (r, g, b) with values from 0 to 1

    Raises:
        ValueError: If the color cannot be parsed
    """
    value_str = value_str.strip()

    # Strip quotes first if it's a quoted string (same logic as safe_parse_value)
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        value_str = value_str[1:-1]

    # Try parsing as a Python literal (for RGB tuples)
    try:
        # If it's already a valid tuple or list, parse it
        color_tuple = ast.literal_eval(value_str)
        if isinstance(color_tuple, (list, tuple)) and len(color_tuple) >= 3:
            # Return just the RGB components as a tuple
            r, g, b = color_tuple[:3]
            return (float(r), float(g), float(b))
    except (SyntaxError, ValueError):
        pass  # Not a valid tuple/list, try other formats

    # Try parsing CSS-style rgb(...) or rgba(...)
    css_rgb_match = re.match(
        r"rgb\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", value_str, re.IGNORECASE
    )
    css_rgba_match = re.match(
        r"rgba\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9\.]+)\s*\)",
        value_str,
        re.IGNORECASE,
    )
    if css_rgb_match:
        r, g, b = map(int, css_rgb_match.groups())
        return (r / 255.0, g / 255.0, b / 255.0)
    elif css_rgba_match:
        r, g, b, _a = css_rgba_match.groups()
        r, g, b = int(r), int(g), int(b)
        # alpha is ignored for now, but could be used if needed
        return (r / 255.0, g / 255.0, b / 255.0)

    # Try as a color name or hex
    try:
        from colour import Color  # type: ignore[import-untyped]

        color = Color(value_str)
        return (float(color.red), float(color.green), float(color.blue))
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Could not parse color value: {value_str}") from e

    # If we got here with a non-tuple, raise error
    raise ValueError(f"Invalid color value: {value_str}")


def _split_top_level_or(selector: str) -> List[str]:
    """
    Split a selector string on top-level OR operators (| or ,) only.

    Respects parsing contexts and does not split when | or , appear inside:
    - Quoted strings (both single and double quotes)
    - Parentheses (for pseudo-class arguments like :not(...))
    - Square brackets (for attribute selectors like [attr="value"])

    Args:
        selector: The selector string to split

    Returns:
        List of selector parts. If no top-level OR operators found, returns [selector].

    Examples:
        >>> _split_top_level_or('text:contains("a|b")|text:bold')
        ['text:contains("a|b")', 'text:bold']

        >>> _split_top_level_or('text:contains("hello,world")')
        ['text:contains("hello,world")']
    """
    if not selector or not isinstance(selector, str):
        return [selector] if selector else []

    parts: List[str] = []
    current_part = ""
    i = 0

    # Parsing state
    in_double_quotes = False
    in_single_quotes = False
    paren_depth = 0
    bracket_depth = 0

    while i < len(selector):
        char = selector[i]

        # Handle escape sequences — count preceding backslashes to handle
        # double-escaped backslashes correctly (e.g. \\" is not an escape).
        if i > 0 and selector[i - 1] == "\\":
            num_slashes = 0
            j = i - 1
            while j >= 0 and selector[j] == "\\":
                num_slashes += 1
                j -= 1
            if num_slashes % 2 == 1:  # odd backslashes → char is escaped
                current_part += char
                i += 1
                continue

        # Handle quote state changes
        if char == '"' and not in_single_quotes:
            in_double_quotes = not in_double_quotes
        elif char == "'" and not in_double_quotes:
            in_single_quotes = not in_single_quotes

        # Handle parentheses and brackets only when not in quotes
        elif not in_double_quotes and not in_single_quotes:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
                if paren_depth < 0:
                    raise ValueError(f"Unmatched closing parenthesis in selector: '{selector}'")
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth < 0:
                    raise ValueError(f"Unmatched closing bracket in selector: '{selector}'")

            # Check for top-level OR operators
            elif (char == "|" or char == ",") and paren_depth == 0 and bracket_depth == 0:
                # Found a top-level OR operator
                part = current_part.strip()
                if part:  # Only add non-empty parts
                    parts.append(part)
                current_part = ""
                i += 1
                continue

        # Add character to current part
        current_part += char
        i += 1

    # Add the final part
    final_part = current_part.strip()
    if final_part:
        parts.append(final_part)

    # If we only found one part, return it as a single-element list
    # If we found multiple parts, those are the OR-separated parts
    return parts if parts else [selector]


# -----------------------------------------------------------------------------
# Selector Parsing Cache
# -----------------------------------------------------------------------------
# LRU cache for parsed selectors. Since parse_selector returns mutable dicts,
# we cache the result and return a deep copy to prevent mutation issues.
# The cache size of 256 is chosen to balance memory usage with cache hit rate
# for typical document processing scenarios.

_SELECTOR_CACHE_SIZE = 256


@functools.lru_cache(maxsize=_SELECTOR_CACHE_SIZE)
def _parse_selector_cached(selector: str) -> Dict[str, Any]:
    """Internal cached parser. Returns cached result - callers must copy if mutating."""
    return _parse_selector_impl(selector)


def clear_selector_cache() -> None:
    """
    Clear the selector parsing cache.

    This can be useful in long-running applications or when memory usage
    is a concern. Under normal usage, the cache is bounded to 256 entries
    and does not need manual clearing.
    """
    _parse_selector_cached.cache_clear()


def get_selector_cache_info():
    """
    Get statistics about the selector parsing cache.

    Returns:
        A named tuple with hits, misses, maxsize, and currsize.

    Example:
        >>> info = get_selector_cache_info()
        >>> print(f"Cache hit rate: {info.hits / (info.hits + info.misses):.1%}")
    """
    return _parse_selector_cached.cache_info()


def parse_selector(selector: str) -> Dict[str, Any]:
    """
    Parse a CSS-like selector string into a structured selector object.

    Handles:
    - Element types (e.g., 'text', 'rect')
    - Attribute presence (e.g., '[data-id]')
    - Attribute value checks with various operators (e.g., '[count=5]', '[name*="bold"]'')
    - Pseudo-classes (e.g., ':contains("Total")', ':empty', ':not(...)')
    - OR operators (e.g., 'text:contains("A")|text:bold', 'sel1,sel2')

    Args:
        selector: CSS-like selector string

    Returns:
        Dict representing the parsed selector, or compound selector with OR logic

    Examples:
        >>> parse_selector('text:contains("hello")')  # Single selector
        {'type': 'text', 'pseudo_classes': [{'name': 'contains', 'args': 'hello'}], ...}

        >>> parse_selector('text:contains("A")|text:bold')  # OR with pipe
        {'type': 'or', 'selectors': [...]}

        >>> parse_selector('text:contains("A"),line[width>5]')  # OR with comma
        {'type': 'or', 'selectors': [...]}

    Note:
        OR operators work with all selector types except spatial pseudo-classes
        (:above, :below, :near, :left-of, :right-of) which require page context.
        Spatial relationships within OR selectors are not currently supported.

        Results are cached for performance. The returned dict is the cached
        object itself and MUST NOT be mutated. Use ``build_execution_plan()``
        to obtain filter functions and post/relational pseudos without mutation.
    """
    return _parse_selector_cached(selector)


def _parse_selector_impl(selector: str) -> Dict[str, Any]:
    """
    Internal implementation of selector parsing.

    This is the actual parser logic, separated from the public API
    to enable caching. Do not call directly - use parse_selector() instead.
    """
    attributes: List[Dict[str, Any]] = []
    pseudo_classes: List[Dict[str, Any]] = []
    filters: List[Dict[str, Any]] = []
    result = {
        "type": "any",
        "attributes": attributes,
        "pseudo_classes": pseudo_classes,
        "filters": filters,
    }

    original_selector_for_error = selector  # Keep for error messages
    if not selector or not isinstance(selector, str):
        return result

    selector = selector.strip()

    # ------------------------------------------------------------------
    # Handle wildcard selector (leading "*")
    # ------------------------------------------------------------------
    # A selector can start with "*" to denote "any element type", optionally
    # followed by attribute blocks or pseudo-classes – e.g. *[width>100].
    # We strip the asterisk but keep the remainder so the normal attribute
    # / pseudo-class parsing logic can proceed.

    if selector.startswith("*"):
        # Keep everything *after* the asterisk (attributes, pseudos, etc.).
        selector = selector[1:].strip()

    # --- Handle OR operators first (| or ,) ---
    # Check if selector contains OR operators at the top level only
    # (not inside quotes, parentheses, or brackets)
    or_parts = _split_top_level_or(selector)

    # If we found OR parts, parse each one recursively and return compound selector
    if len(or_parts) > 1:
        # Deduplicate OR parts while preserving order (first occurrence wins)
        # This optimization avoids parsing and evaluating the same selector multiple times
        seen = set()
        unique_parts = []
        for part in or_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        # Use cached parser for recursive calls to avoid redundant parsing
        # Note: We use _parse_selector_cached directly here since we're building
        # a new compound result that will be cached at the top level
        parsed_selectors = [_parse_selector_cached(part) for part in unique_parts]

        if len(parsed_selectors) > 1:
            return {"type": "or", "selectors": parsed_selectors}
        elif len(parsed_selectors) == 1:
            # Only one valid part after deduplication, return it directly
            return parsed_selectors[0]
        else:
            raise ValueError(f"No valid parts found in OR selector '{original_selector_for_error}'")

    # --- Continue with single selector parsing (existing logic) ---

    # 1. Extract type (optional, at the beginning)
    # Only run if selector wasn't '*'
    if selector:
        type_match = re.match(r"^([a-zA-Z_\-]+)", selector)
        if type_match:
            result["type"] = type_match.group(1).lower()
            selector = selector[len(type_match.group(0)) :].strip()

    # Regexes for parts at the START of the remaining string
    # Attribute: Starts with [, ends with ], content is non-greedy non-] chars
    attr_pattern = re.compile(r"^\[\s*([^\s\]]+.*?)\s*\]")
    # Pseudo: Starts with :, name is letters/hyphen/underscore, optionally followed by (...)
    pseudo_pattern = re.compile(r"^:([a-zA-Z_\-]+)(?:\((.*?)\))?")
    # :not() specifically requires careful parenthesis matching later
    not_pseudo_prefix = ":not("

    # 2. Iteratively parse attributes and pseudo-classes
    while selector:
        processed_chunk = False

        # Check for attribute block `[...]`
        attr_match = attr_pattern.match(selector)
        if attr_match:
            block_content = attr_match.group(1).strip()
            # Parse the content inside the block
            # Pattern: name, optional op, optional value
            detail_match = re.match(
                r"^([a-zA-Z0-9_\-]+)\s*(?:(>=|<=|>|<|!=|[\*\~\^\$]?=)\s*(.*?))?$", block_content
            )
            if not detail_match:
                raise ValueError(
                    f"Invalid attribute syntax inside block: '[{block_content}]'. Full selector: '{original_selector_for_error}'"
                )

            name, op, value_str = detail_match.groups()

            if op is None:
                # Presence selector [attr]
                attributes.append({"name": name, "op": "exists", "value": None})
            else:
                # Operator exists, value must also exist (even if empty via quotes)
                if value_str is None:  # Catches invalid [attr=]
                    raise ValueError(
                        f"Invalid selector: Attribute '[{name}{op}]' must have a value. Use '[{name}{op}\"\"]' for empty string or '[{name}]' for presence. Full selector: '{original_selector_for_error}'"
                    )
                # Parse value - check for aggregate functions first
                parsed_value: Any
                aggregate_func = _parse_aggregate_function(value_str)

                if aggregate_func:
                    # Store aggregate function info
                    parsed_value = aggregate_func
                elif name in [
                    "color",
                    "non_stroking_color",
                    "fill",
                    "stroke",
                    "strokeColor",
                    "fillColor",
                ]:
                    parsed_value = safe_parse_color(value_str)
                else:
                    parsed_value = safe_parse_value(value_str)  # Handles quotes
                    # If using ~= with a numeric value, warn once during parsing
                    if op == "~=" and isinstance(parsed_value, (int, float)):
                        raise ValueError(
                            "Numeric comparison with '~=' is ambiguous. "
                            "Use explicit ranges such as [width>1][width<4]."
                        )
                attributes.append({"name": name, "op": op, "value": parsed_value})

            selector = selector[attr_match.end() :].strip()
            processed_chunk = True
            continue

        # Check for :not(...) block
        if selector.lower().startswith(not_pseudo_prefix):
            start_index = len(not_pseudo_prefix) - 1  # Index of '('
            nesting = 1
            end_index = -1
            for i in range(start_index + 1, len(selector)):
                if selector[i] == "(":
                    nesting += 1
                elif selector[i] == ")":
                    nesting -= 1
                    if nesting == 0:
                        end_index = i
                        break

            if end_index == -1:
                raise ValueError(
                    f"Mismatched parenthesis in :not() selector near '{selector}'. Full selector: '{original_selector_for_error}'"
                )

            inner_selector_str = selector[start_index + 1 : end_index].strip()
            if not inner_selector_str:
                raise ValueError(
                    f"Empty selector inside :not(). Full selector: '{original_selector_for_error}'"
                )

            # Recursively parse the inner selector
            parsed_inner_selector = parse_selector(inner_selector_str)
            pseudo_classes.append({"name": "not", "args": parsed_inner_selector})

            selector = selector[end_index + 1 :].strip()
            processed_chunk = True
            continue

        # Check for other pseudo-class blocks `:name` or `:name(...)`
        pseudo_match = pseudo_pattern.match(selector)
        if pseudo_match:
            # --- NEW: robustly capture arguments that may contain nested parentheses --- #
            name, args_str = pseudo_match.groups()
            match_end_idx = pseudo_match.end()

            # If the args_str has unbalanced parens (respecting quotes) or unbalanced
            # quotes, extend the capture past the initial regex match until everything
            # is balanced. This handles :contains((Tre) Ofertu), :regex("foo(bar)"),
            # and other complex patterns.
            if args_str is not None:
                # Check if the initial regex capture is correct by verifying
                # that quotes and parens are balanced within args_str.
                balance = 0
                _in_dq = False
                _in_sq = False
                for ci, ch in enumerate(args_str):
                    prev_ch = args_str[ci - 1] if ci > 0 else ""
                    if prev_ch != "\\":
                        if ch == '"' and not _in_sq:
                            _in_dq = not _in_dq
                        elif ch == "'" and not _in_dq:
                            _in_sq = not _in_sq
                        elif not _in_dq and not _in_sq:
                            if ch == "(":
                                balance += 1
                            elif ch == ")":
                                balance -= 1

                # If the initial regex capture got confused by quotes or nested
                # parens (e.g. :regex("foo(bar)") where non-greedy .*? stops at
                # the wrong ")"), rescan from the opening "(" with full
                # quote/paren awareness to find the correct closing ")".
                if balance > 0 or _in_dq or _in_sq:
                    open_paren_pos = 1 + len(name)  # position of '(' in selector
                    args_start = open_paren_pos + 1
                    depth = 1
                    in_dq = False
                    in_sq = False
                    i = args_start
                    while i < len(selector):
                        char = selector[i]
                        prev_char = selector[i - 1] if i > 0 else ""
                        if prev_char != "\\":
                            if char == '"' and not in_sq:
                                in_dq = not in_dq
                            elif char == "'" and not in_dq:
                                in_sq = not in_sq
                            elif not in_dq and not in_sq:
                                if char == "(":
                                    depth += 1
                                elif char == ")":
                                    depth -= 1
                                    if depth == 0:
                                        break
                        i += 1
                    if depth != 0:
                        raise ValueError(
                            f"Mismatched parentheses or quotes in pseudo-class :{name}(). Full selector: '{original_selector_for_error}'"
                        )
                    args_str = selector[args_start:i]
                    match_end_idx = i + 1  # past the closing ')'

            name = name.lower()  # Normalize pseudo-class name
            processed_args = args_str  # Keep as string initially, or None

            if args_str is not None:
                # Only parse args if they exist and based on the pseudo-class type
                if name in ["color", "background"]:
                    processed_args = safe_parse_color(args_str)
                else:
                    processed_args = safe_parse_value(args_str)
            # else: args remain None

            pseudo_classes.append({"name": name, "args": processed_args})
            # IMPORTANT: use match_end_idx (may have been extended)
            selector = selector[match_end_idx:].strip()
            processed_chunk = True
            continue

        # If we reach here and the selector string is not empty, something is wrong
        if not processed_chunk and selector:
            raise ValueError(
                f"Invalid or unexpected syntax near '{selector[:30]}...'. Full selector: '{original_selector_for_error}'"
            )

    return result


def _is_color_value(value) -> bool:
    """
    Check if a value represents a color by attempting to parse it with Color.
    """
    try:
        # If it's already a tuple/list, convert to tuple
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return True
        # Otherwise try parsing as a color name/hex
        from colour import Color  # type: ignore[import-untyped]

        Color(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _extract_rgb_triplet(value: Any) -> tuple[float, float, float]:
    """Normalize various color inputs into an RGB triplet."""
    if isinstance(value, (list, tuple)):
        if len(value) < 3:
            raise ValueError("Color tuples must include at least three components.")
        r, g, b = value[0], value[1], value[2]
    else:
        from colour import Color  # type: ignore[import-untyped]

        rgb_sequence = cast(Sequence[float], Color(value).rgb)
        r, g, b = rgb_sequence[0], rgb_sequence[1], rgb_sequence[2]
    return float(r), float(g), float(b)


def _color_distance(color1: Any, color2: Any) -> float:
    """
    Calculate Delta E color difference between two colors.
    Colors can be strings (names/hex) or RGB tuples.

    Returns:
        Delta E value, or float('inf') if colors can't be compared
    """
    try:
        from colormath2.color_conversions import convert_color  # type: ignore[import-untyped]
        from colormath2.color_diff import delta_e_cie2000  # type: ignore[import-untyped]
        from colormath2.color_objects import LabColor, sRGBColor  # type: ignore[import-untyped]

        # Convert to RGB tuples
        rgb1 = sRGBColor(*_extract_rgb_triplet(color1))
        rgb2 = sRGBColor(*_extract_rgb_triplet(color2))

        lab1 = convert_color(rgb1, LabColor)
        lab2 = convert_color(rgb2, LabColor)
        return float(delta_e_cie2000(lab1, lab2))
    except (ValueError, TypeError, AttributeError):
        return float("inf")


def _is_approximate_match(value1: Any, value2: Any) -> bool:
    """
    Check if two values approximately match.

    For colors: Uses Delta E color difference with tolerance of 20.0
    For numbers: Uses absolute difference with tolerance of 2.0
    """
    # First check if both values are colors
    if _is_color_value(value1) and _is_color_value(value2):
        return bool(_color_distance(value1, value2) <= 20.0)

    # Then check if both are numbers
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return bool(abs(value1 - value2) <= 2.0)

    # Default to exact match for other types
    return bool(value1 == value2)


def _is_exact_color_match(value1, value2) -> bool:
    """
    Check if two color values match exactly (with small tolerance for color variations).

    For colors: Uses Delta E color difference with strict tolerance of 2.0
    For non-colors: Falls back to exact equality
    """
    # First check if both values are colors
    if _is_color_value(value1) and _is_color_value(value2):
        return _color_distance(value1, value2) <= 2.0

    # Default to exact match for non-colors
    return bool(value1 == value2)


def _build_filter_list(
    selector: Dict[str, Any], aggregates: Optional[Dict[str, Any]] = None, **kwargs
) -> List[Dict[str, Any]]:
    """
    Convert a parsed selector to a list of named filter functions.

    Args:
        selector: Parsed selector dictionary
        aggregates: Pre-calculated aggregate values (optional)
        **kwargs: Additional filter parameters including:
                 - regex: Whether to use regex for text search
                 - case: Whether to do case-sensitive text search

    Returns:
        Tuple of (filters, post_pseudos, relational_pseudos):
        - filters: List of dicts with 'name' (str) and 'func' (callable) for element-level matching
        - post_pseudos: List of pseudo dicts for collection-level operations (:first, :last, :nth, etc.)
        - relational_pseudos: List of pseudo dicts for spatial operations (:above, :below, :near, etc.)
    """
    filters: List[Dict[str, Any]] = []
    selector_type = selector["type"]
    attribute_filters: List[Dict[str, Any]] = list(selector.get("attributes", []))
    pseudo_entries: List[Dict[str, Any]] = list(selector.get("pseudo_classes", []))

    if aggregates is None:
        aggregate_values: Dict[str, Any] = {}
    else:
        aggregate_values = dict(aggregates)

    clause_ctx = ClauseEvalContext(
        selector_context=kwargs.get("selector_context"),
        aggregates=aggregate_values,
        options=kwargs,
    )

    post_pseudos: List[Dict[str, Any]] = []
    relational_pseudos: List[Dict[str, Any]] = []

    def _extend_from_handler(result: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]):
        if not result:
            return
        if isinstance(result, list):
            filters.extend(result)
        else:
            filters.append(result)

    # Filter by element type
    if selector_type != "any":
        filter_name = f"type is '{selector_type}'"
        normalized_selector_type = _normalize_selector_label(selector_type)
        if selector_type == "text":
            filter_name = "type is 'text', 'char', or 'word'"
            func = lambda el: hasattr(el, "type") and el.type in ["text", "char", "word"]
        elif selector_type == "region":
            filter_name = "type is 'region' (has region_type)"
            # Note: Specific region type attribute (e.g., [type=table]) is checked below
            func = lambda el: hasattr(el, "region_type")
        else:
            # Check against normalized_type first, then element.type
            func = lambda el: (
                hasattr(el, "normalized_type")
                and _normalize_selector_label(el.normalized_type) == normalized_selector_type
            ) or (
                not hasattr(
                    el, "normalized_type"
                )  # Only check element.type if normalized_type doesn't exist/match
                and hasattr(el, "type")
                and _normalize_selector_label(el.type) == normalized_selector_type
            )
        filters.append({"name": filter_name, "func": func})

    def _make_exists_filter(getter: Callable[[Any], Any]) -> Callable[[Any], bool]:
        def exists_filter(element: Any) -> bool:
            return getter(element) is not None

        return exists_filter

    def _make_compare_filter(
        getter: Callable[[Any], Any],
        comparator: Callable[[Any, Any], bool],
        expected_value: Any,
        attr_name: str,
    ) -> Callable[[Any], bool]:
        def compare_filter(element: Any) -> bool:
            value = getter(element)
            if value is None:
                return False
            try:
                return bool(comparator(value, expected_value))
            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "Comparison failed for attribute '%s': %s", attr_name, exc, exc_info=True
                )
                return False

        return compare_filter

    # Filter by attributes
    for attr_filter in attribute_filters:
        name = attr_filter["name"]
        op = attr_filter["op"]
        value = attr_filter["value"]
        python_name = name.replace("-", "_")  # Convert CSS-style names

        handler = get_attribute_handler(name)
        if handler:
            handler_result = handler(attr_filter, clause_ctx)
            _extend_from_handler(handler_result)
            continue

        # Check if value is an aggregate function
        if isinstance(value, dict) and value.get("type") == "aggregate":
            # Use pre-calculated aggregate value
            aggregate_value = aggregate_values.get(name)
            if aggregate_value is None:
                # Aggregate couldn't be calculated; ensure no elements match this filter
                filters.append(
                    {
                        "name": f"aggregate {value['func']} for '{name}' unavailable",
                        "func": lambda _el: False,
                    }
                )
                continue

            # Apply arithmetic operation if specified
            if "operator" in value and "operand" in value:
                operator = value["operator"]
                operand = value["operand"]

                try:
                    if operator == "+":
                        aggregate_value = aggregate_value + operand
                    elif operator == "-":
                        aggregate_value = aggregate_value - operand
                    elif operator == "*":
                        aggregate_value = aggregate_value * operand
                    elif operator == "/":
                        if operand == 0:
                            raise ZeroDivisionError(
                                f"Division by zero in aggregate expression for '{name}'"
                            )
                        aggregate_value = aggregate_value / operand
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Could not apply arithmetic operation to aggregate value for '{name}': {e}"
                    ) from e

            value = aggregate_value

        # --- Define the core value retrieval logic ---
        def get_element_value(
            element: Any, *, attr_name: str = name, py_name: str = python_name
        ) -> Any:
            bbox_mapping = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
            if attr_name in bbox_mapping:
                bbox = getattr(element, "_bbox", None) or getattr(element, "bbox", None)
                if bbox is None:
                    return None
                return bbox[bbox_mapping[attr_name]]

            # Special case for region attributes
            if selector_type == "region":
                if attr_name == "type":
                    if hasattr(element, "normalized_type") and element.normalized_type:
                        return _normalize_selector_label(element.normalized_type)
                    else:
                        return _normalize_selector_label(getattr(element, "region_type", ""))
                elif attr_name == "model":
                    return getattr(element, "model", None)
                elif attr_name == "checked":
                    # Map 'checked' attribute to is_checked for checkboxes
                    return getattr(element, "is_checked", None)
                else:
                    return getattr(element, py_name, None)
            else:
                # General case for non-region elements
                return getattr(element, py_name, None)

        if op == "exists":
            filter_label = f"attribute [{name} exists]"
            filters.append({"name": filter_label, "func": _make_exists_filter(get_element_value)})
            continue

        # Handle operators with values (e.g., =, !=, *=, etc.)
        compare_func: Callable[[Any, Any], bool]
        op_desc = f"{op} {value!r}"  # Default description

        if op == "=":
            if name in [
                "color",
                "non_stroking_color",
                "fill",
                "stroke",
                "strokeColor",
                "fillColor",
            ]:

                def compare_color(el_val: Any, sel_val: Any) -> bool:
                    return _is_exact_color_match(el_val, sel_val)

                compare_func = compare_color
                op_desc = f"= {value!r} (exact color)"
            elif name in ["checked", "is_checked", "bold", "italic"]:

                def compare_bool(el_val: Any, sel_val: Any) -> bool:
                    el_bool = (
                        el_val
                        if isinstance(el_val, bool)
                        else str(el_val).lower()
                        in (
                            "true",
                            "1",
                            "yes",
                        )
                    )
                    sel_bool = (
                        sel_val
                        if isinstance(sel_val, bool)
                        else str(sel_val).lower()
                        in (
                            "true",
                            "1",
                            "yes",
                        )
                    )
                    return el_bool == sel_bool

                compare_func = compare_bool
            else:

                def compare_equal(el_val: Any, sel_val: Any) -> bool:
                    return bool(el_val == sel_val)

                compare_func = compare_equal
        elif op == "!=":

            def compare_not_equal(el_val: Any, sel_val: Any) -> bool:
                return bool(el_val != sel_val)

            compare_func = compare_not_equal
        elif op == "~=":
            if isinstance(value, (int, float)):
                tolerance = abs(value) * 0.1

                def compare_numeric_tolerance(el_val: Any, sel_val: Any) -> bool:
                    return (
                        isinstance(el_val, (int, float))
                        and isinstance(sel_val, (int, float))
                        and (sel_val - tolerance) <= el_val <= (sel_val + tolerance)
                    )

                compare_func = compare_numeric_tolerance
                op_desc = f"~= {value!r} (±10%)"
                value = float(value)
            else:
                op_desc = f"~= {value!r} (approx)"

                def compare_approx(el_val: Any, sel_val: Any) -> bool:
                    return _is_approximate_match(el_val, sel_val)

                compare_func = compare_approx
        elif op == "^=":

            def compare_prefix(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, str)
                    and isinstance(sel_val, str)
                    and el_val.startswith(sel_val)
                )

            compare_func = compare_prefix
        elif op == "$=":

            def compare_suffix(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, str)
                    and isinstance(sel_val, str)
                    and el_val.endswith(sel_val)
                )

            compare_func = compare_suffix
        elif op == "*=":
            if name == "fontname":
                op_desc = f"*= {value!r} (contains, case-insensitive)"

                def compare_contains_ci(el_val: Any, sel_val: Any) -> bool:
                    return (
                        isinstance(el_val, str)
                        and isinstance(sel_val, str)
                        and sel_val.lower() in el_val.lower()
                    )

                compare_func = compare_contains_ci
            else:
                op_desc = f"*= {value!r} (contains)"

                def compare_contains(el_val: Any, sel_val: Any) -> bool:
                    return (
                        isinstance(el_val, str) and isinstance(sel_val, str) and sel_val in el_val
                    )

                compare_func = compare_contains
        elif op == ">=":

            def compare_ge(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val >= sel_val
                )

            compare_func = compare_ge
        elif op == "<=":

            def compare_le(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val <= sel_val
                )

            compare_func = compare_le
        elif op == ">":

            def compare_gt(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val > sel_val
                )

            compare_func = compare_gt
        elif op == "<":

            def compare_lt(el_val: Any, sel_val: Any) -> bool:
                return (
                    isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val < sel_val
                )

            compare_func = compare_lt
        else:
            raise ValueError(f"Unsupported operator '{op}' encountered for attribute '{name}'")

        filter_label = f"attribute [{name}{op_desc}]"
        filters.append(
            {
                "name": filter_label,
                "func": _make_compare_filter(get_element_value, compare_func, value, name),
            }
        )

    # Filter by pseudo-classes
    for pseudo in pseudo_entries:
        name = pseudo["name"]
        args = pseudo["args"]

        # Relational pseudo-classes and collection-level pseudo-classes are handled separately by the caller

        handler = get_pseudo_handler(name)
        if handler:
            handler_result = handler(pseudo, clause_ctx)
            _extend_from_handler(handler_result)
            continue
        if get_post_handler(name):
            post_pseudos.append(pseudo)
            continue
        if get_relational_handler(name):
            relational_pseudos.append(pseudo)
            continue

        # --- Handle :not() ---
        elif name == "not":
            if not isinstance(args, dict):  # args should be the parsed inner selector
                logger.error(f"Invalid arguments for :not pseudo-class: {args}")
                raise TypeError(
                    "Internal error: :not pseudo-class requires a parsed selector dictionary as args."
                )

            # Recursively get the filter function for the inner selector
            # Pass kwargs and aggregates down in case regex/case flags affect the inner selector
            inner_filter_func = selector_to_filter_func(args, aggregates=aggregate_values, **kwargs)

            def not_filter(
                element: Any, inner_func: Callable[[Any], bool] = inner_filter_func
            ) -> bool:
                return not inner_func(element)

            inner_filter_list, _, _ = _build_filter_list(
                args, aggregates=aggregate_values, **kwargs
            )
            inner_filter_names = ", ".join([f["name"] for f in inner_filter_list])
            filter_name = f"pseudo-class :not({inner_filter_names})"
            filters.append({"name": filter_name, "func": not_filter})
            continue

        # --- Handle :closest pseudo-class for fuzzy text matching --- #
        elif name == "closest" and args is not None:
            # Note: :closest is handled specially in the page._apply_selector method
            # It doesn't filter elements here, but marks them for special processing
            # This allows us to first check :contains matches, then sort by similarity
            def closest_filter(_element: Any) -> bool:
                return True

            filters.append({"name": "pseudo-class :closest", "func": closest_filter})
            continue

        # --- Handle :ocr pseudo-class for OCR-tolerant matching --- #
        elif name == "ocr" and args is not None:
            # Like :closest, handled specially in page._apply_selector
            def ocr_filter(_element: Any) -> bool:
                return True

            filters.append({"name": "pseudo-class :ocr", "func": ocr_filter})
            continue

        else:
            raise ValueError(f"Unknown or unsupported pseudo-class: ':{name}'")

    return filters, post_pseudos, relational_pseudos


def _assemble_filter_func(filters: List[Dict[str, Any]]) -> Callable[[Any], bool]:
    """
    Combine a list of named filter functions into a single callable.

    Args:
        filters: List of dictionaries, each with 'name' and 'func'.

    Returns:
        A single function that takes an element and returns True only if
        it passes ALL filters in the list.
    """

    def combined_filter(element):
        for f in filters:
            if not f["func"](element):
                return False
        return True

    return combined_filter


def _calculate_aggregates(elements: List[Any], selector: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate values for a selector.

    Args:
        elements: List of elements to calculate aggregates from
        selector: Parsed selector dictionary

    Returns:
        Dict mapping attribute names to their aggregate values
    """
    aggregates: Dict[str, Any] = {}

    # Find all aggregate functions in attributes
    for attr in selector.get("attributes", []):
        value = attr.get("value")
        if isinstance(value, dict) and value.get("type") == "aggregate":
            attr_name = attr["name"]
            func_name = value["func"]
            func_args = value.get("args")

            # Extract attribute values from elements
            values = []
            for el in elements:
                # Handle special bbox attributes
                if attr_name in ["x0", "y0", "x1", "y1"]:
                    bbox_mapping = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
                    bbox = getattr(el, "_bbox", None) or getattr(el, "bbox", None)
                    try:
                        if bbox:
                            val = bbox[bbox_mapping[attr_name]]
                            values.append(val)
                    except (TypeError, IndexError):
                        continue
                else:
                    val = getattr(el, attr_name.replace("-", "_"), None)
                    if val is not None:
                        values.append(val)

            if not values:
                # No valid values found, aggregate is None
                aggregates[attr_name] = None
                continue

            # Calculate aggregate based on function
            if func_name == "min":
                aggregates[attr_name] = min(values)
            elif func_name == "max":
                aggregates[attr_name] = max(values)
            elif func_name == "avg":
                try:
                    aggregates[attr_name] = sum(values) / len(values)
                except TypeError:
                    # Non-numeric values
                    aggregates[attr_name] = None
            elif func_name == "median":
                try:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    if n % 2 == 0:
                        aggregates[attr_name] = (
                            sorted_values[n // 2 - 1] + sorted_values[n // 2]
                        ) / 2
                    else:
                        aggregates[attr_name] = sorted_values[n // 2]
                except TypeError:
                    # Non-numeric values
                    aggregates[attr_name] = None
            elif func_name == "mode":
                # Works for any type
                counter = Counter(values)
                most_common = counter.most_common(1)
                if most_common:
                    aggregates[attr_name] = most_common[0][0]
                else:
                    aggregates[attr_name] = None
            elif func_name == "closest" and func_args is not None:
                # For colors, find the value with minimum distance
                if attr_name in [
                    "color",
                    "non_stroking_color",
                    "fill",
                    "stroke",
                    "strokeColor",
                    "fillColor",
                ]:
                    min_distance = float("inf")
                    closest_value = None
                    for val in values:
                        try:
                            distance = _color_distance(val, func_args)
                            if distance < min_distance:
                                min_distance = distance
                                closest_value = val
                        except:
                            continue
                    aggregates[attr_name] = closest_value
                else:
                    # For non-colors, closest doesn't make sense
                    aggregates[attr_name] = None

    return aggregates


def build_execution_plan(
    selector: Dict[str, Any], aggregates: Optional[Dict[str, Any]] = None, **kwargs
) -> Tuple[Callable[[Any], bool], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a complete execution plan from a parsed selector without mutating it.

    Returns:
        Tuple of (filter_func, post_pseudos, relational_pseudos).
        - filter_func: callable that takes an element and returns True if it matches
        - post_pseudos: collection-level pseudos like :first, :last, :nth, :slice, :limit
        - relational_pseudos: spatial pseudos like :above, :below, :near
    """
    if selector.get("type") == "or":
        sub_selectors = selector.get("selectors", [])
        if not sub_selectors:
            return (lambda element: False), [], []

        sub_filter_funcs = []
        all_post: List[Dict[str, Any]] = []
        all_relational: List[Dict[str, Any]] = []
        for sub_selector in sub_selectors:
            sub_func, sub_post, sub_relational = build_execution_plan(
                sub_selector, aggregates=aggregates, **kwargs
            )
            sub_filter_funcs.append(sub_func)
            all_post.extend(sub_post)
            all_relational.extend(sub_relational)

        def or_filter(element):
            return any(func(element) for func in sub_filter_funcs)

        return or_filter, all_post, all_relational

    filter_list, post_pseudos, relational_pseudos = _build_filter_list(
        selector, aggregates=aggregates, **kwargs
    )
    filter_func = _assemble_filter_func(filter_list)
    return filter_func, post_pseudos, relational_pseudos


def selector_to_filter_func(
    selector: Dict[str, Any], aggregates: Optional[Dict[str, Any]] = None, **kwargs
) -> Callable[[Any], bool]:
    """
    Convert a parsed selector to a single filter function.

    For callers that also need post/relational pseudos, use ``build_execution_plan`` instead.
    """
    filter_func, _, _ = build_execution_plan(selector, aggregates=aggregates, **kwargs)
    return filter_func
