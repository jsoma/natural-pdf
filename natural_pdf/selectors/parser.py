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
import difflib
import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from colormath2.color_conversions import convert_color
from colormath2.color_diff import delta_e_cie2000
from colormath2.color_objects import LabColor, sRGBColor
from colour import Color

logger = logging.getLogger(__name__)


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

    Returns:
        Dict with 'type': 'aggregate', 'func': function name, 'args': optional args
        or None if not an aggregate function.
    """
    value_str = value_str.strip()

    # Pattern for aggregate functions: funcname() or funcname(args)
    # Supports: min(), max(), avg(), mean(), median(), mode(), most_common(), closest(...)
    func_pattern = re.match(
        r"^(min|max|avg|mean|median|mode|most_common|closest)\s*\((.*?)\)$",
        value_str,
        re.IGNORECASE,
    )

    if not func_pattern:
        return None

    func_name = func_pattern.group(1).lower()
    args_str = func_pattern.group(2).strip()

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

    return {"type": "aggregate", "func": func_name, "args": args}


def safe_parse_color(value_str: str) -> tuple:
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
            return tuple(color_tuple[:3])
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
        r, g, b, a = css_rgba_match.groups()
        r, g, b = int(r), int(g), int(b)
        # alpha is ignored for now, but could be used if needed
        return (r / 255.0, g / 255.0, b / 255.0)

    # Try as a color name or hex
    try:
        color = Color(value_str)
        return (color.red, color.green, color.blue)
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

    parts = []
    current_part = ""
    i = 0

    # Parsing state
    in_double_quotes = False
    in_single_quotes = False
    paren_depth = 0
    bracket_depth = 0

    while i < len(selector):
        char = selector[i]

        # Handle escape sequences in quotes
        if i > 0 and selector[i - 1] == "\\":
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
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1

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
    """
    result = {
        "type": "any",
        "attributes": [],
        "pseudo_classes": [],
        "filters": [],  # Keep this for potential future use
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
        parsed_selectors = []
        for part in or_parts:
            try:
                parsed_selectors.append(parse_selector(part))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid OR selector part '{part}': {e}")
                continue

        if len(parsed_selectors) > 1:
            return {"type": "or", "selectors": parsed_selectors}
        elif len(parsed_selectors) == 1:
            # Only one valid part, return it directly
            return parsed_selectors[0]
        else:
            # No valid parts, return default
            logger.warning(
                f"No valid parts found in OR selector '{original_selector_for_error}', returning default selector"
            )
            return result

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
                result["attributes"].append({"name": name, "op": "exists", "value": None})
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
                        logger.warning(
                            f"Using ~= with numeric values. This will match if the absolute difference is <= 2.0. "
                            f"Consider using explicit ranges (e.g., [width>1][width<4]) for more control."
                        )
                result["attributes"].append({"name": name, "op": op, "value": parsed_value})

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
            result["pseudo_classes"].append({"name": "not", "args": parsed_inner_selector})

            selector = selector[end_index + 1 :].strip()
            processed_chunk = True
            continue

        # Check for other pseudo-class blocks `:name` or `:name(...)`
        pseudo_match = pseudo_pattern.match(selector)
        if pseudo_match:
            # --- NEW: robustly capture arguments that may contain nested parentheses --- #
            name, args_str = pseudo_match.groups()
            match_end_idx = pseudo_match.end()

            # If the args_str contains unmatched opening parens, continue scanning the
            # selector until parentheses are balanced. This allows patterns like
            # :contains((Tre) Ofertu) or complex regex with grouping.
            if args_str is not None and args_str.count("(") > args_str.count(")"):
                balance = args_str.count("(") - args_str.count(")")
                i = match_end_idx
                while i < len(selector) and balance > 0:
                    char = selector[i]
                    # Append char to args_str as we extend the capture
                    args_str += char
                    if char == "(":
                        balance += 1
                    elif char == ")":
                        balance -= 1
                    i += 1
                # After loop, ensure parentheses are balanced; otherwise raise error
                if balance != 0:
                    raise ValueError(
                        f"Mismatched parentheses in pseudo-class :{name}(). Full selector: '{original_selector_for_error}'"
                    )
                # Update where the selector should be sliced off from
                match_end_idx = i

            name = name.lower()  # Normalize pseudo-class name
            processed_args = args_str  # Keep as string initially, or None

            if args_str is not None:
                # Only parse args if they exist and based on the pseudo-class type
                if name in ["color", "background"]:
                    processed_args = safe_parse_color(args_str)
                else:
                    processed_args = safe_parse_value(args_str)
            # else: args remain None

            result["pseudo_classes"].append({"name": name, "args": processed_args})
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
        Color(value)
        return True
    except:
        return False


def _color_distance(color1, color2) -> float:
    """
    Calculate Delta E color difference between two colors.
    Colors can be strings (names/hex) or RGB tuples.

    Returns:
        Delta E value, or float('inf') if colors can't be compared
    """
    try:
        # Convert to RGB tuples
        if isinstance(color1, (list, tuple)) and len(color1) >= 3:
            rgb1 = sRGBColor(*color1[:3])
        else:
            rgb1 = sRGBColor(*Color(color1).rgb)

        if isinstance(color2, (list, tuple)) and len(color2) >= 3:
            rgb2 = sRGBColor(*color2[:3])
        else:
            rgb2 = sRGBColor(*Color(color2).rgb)

        lab1 = convert_color(rgb1, LabColor)
        lab2 = convert_color(rgb2, LabColor)
        return delta_e_cie2000(lab1, lab2)
    except:
        return float("inf")


def _is_approximate_match(value1, value2) -> bool:
    """
    Check if two values approximately match.

    For colors: Uses Delta E color difference with tolerance of 20.0
    For numbers: Uses absolute difference with tolerance of 2.0
    """
    # First check if both values are colors
    if _is_color_value(value1) and _is_color_value(value2):
        return _color_distance(value1, value2) <= 20.0

    # Then check if both are numbers
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return abs(value1 - value2) <= 2.0

    # Default to exact match for other types
    return value1 == value2


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
    return value1 == value2


PSEUDO_CLASS_FUNCTIONS = {
    "bold": lambda el: hasattr(el, "bold") and el.bold,
    "italic": lambda el: hasattr(el, "italic") and el.italic,
    "first-child": lambda el: hasattr(el, "parent") and el.parent and el.parent.children[0] == el,
    "last-child": lambda el: hasattr(el, "parent") and el.parent and el.parent.children[-1] == el,
    "empty": lambda el: not hasattr(el, "text") or not el.text or not el.text.strip(),
    "not-empty": lambda el: bool(hasattr(el, "text") and el.text and el.text.strip()),
    "not-bold": lambda el: hasattr(el, "bold") and not el.bold,
    "not-italic": lambda el: hasattr(el, "italic") and not el.italic,
}


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
        List of dictionaries, each with 'name' (str) and 'func' (callable).
        The callable takes an element and returns True if it matches the specific filter.
    """
    filters: List[Dict[str, Any]] = []
    selector_type = selector["type"]

    if aggregates is None:
        aggregates = {}

    # Filter by element type
    if selector_type != "any":
        filter_name = f"type is '{selector_type}'"
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
                hasattr(el, "normalized_type") and el.normalized_type == selector_type
            ) or (
                not hasattr(
                    el, "normalized_type"
                )  # Only check element.type if normalized_type doesn't exist/match
                and hasattr(el, "type")
                and el.type == selector_type
            )
        filters.append({"name": filter_name, "func": func})

    # Filter by attributes
    for attr_filter in selector["attributes"]:
        name = attr_filter["name"]
        op = attr_filter["op"]
        value = attr_filter["value"]
        python_name = name.replace("-", "_")  # Convert CSS-style names

        # Check if value is an aggregate function
        if isinstance(value, dict) and value.get("type") == "aggregate":
            # Use pre-calculated aggregate value
            aggregate_value = aggregates.get(name)
            if aggregate_value is None:
                # Skip this filter if aggregate couldn't be calculated
                continue
            value = aggregate_value

        # --- Define the core value retrieval logic ---
        def get_element_value(
            element, name=name, python_name=python_name, selector_type=selector_type
        ):
            bbox_mapping = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
            if name in bbox_mapping:
                bbox = getattr(element, "_bbox", None) or getattr(element, "bbox", None)
                return bbox[bbox_mapping[name]]

            # Special case for region attributes
            if selector_type == "region":
                if name == "type":
                    if hasattr(element, "normalized_type") and element.normalized_type:
                        return element.normalized_type
                    else:
                        return getattr(element, "region_type", "").lower().replace(" ", "_")
                elif name == "model":
                    return getattr(element, "model", None)
                elif name == "checked":
                    # Map 'checked' attribute to is_checked for checkboxes
                    return getattr(element, "is_checked", None)
                else:
                    return getattr(element, python_name, None)
            else:
                # General case for non-region elements
                return getattr(element, python_name, None)

        # --- Define the comparison function or direct check ---
        filter_lambda: Callable[[Any], bool]
        filter_name: str

        if op == "exists":
            # Special handling for attribute presence check [attr]
            filter_name = f"attribute [{name} exists]"
            # Lambda checks that the retrieved value is not None
            filter_lambda = lambda el, get_val=get_element_value: get_val(el) is not None
        else:
            # Handle operators with values (e.g., =, !=, *=, etc.)
            compare_func: Callable[[Any, Any], bool]
            op_desc = f"{op} {value!r}"  # Default description

            # Determine compare_func based on op (reuse existing logic)
            if op == "=":
                # For color attributes, use exact color matching with small tolerance
                if name in [
                    "color",
                    "non_stroking_color",
                    "fill",
                    "stroke",
                    "strokeColor",
                    "fillColor",
                ]:
                    op_desc = f"= {value!r} (exact color)"
                    compare_func = lambda el_val, sel_val: _is_exact_color_match(el_val, sel_val)
                # For boolean attributes, handle string/bool comparison
                elif name in ["checked", "is_checked", "bold", "italic"]:

                    def bool_compare(el_val, sel_val):
                        # Convert both to boolean for comparison
                        if isinstance(el_val, bool):
                            el_bool = el_val
                        else:
                            el_bool = str(el_val).lower() in ("true", "1", "yes")

                        if isinstance(sel_val, bool):
                            sel_bool = sel_val
                        else:
                            sel_bool = str(sel_val).lower() in ("true", "1", "yes")

                        # Debug logging
                        logger.debug(
                            f"Boolean comparison: el_val={el_val} ({type(el_val)}) -> {el_bool}, sel_val={sel_val} ({type(sel_val)}) -> {sel_bool}"
                        )

                        return el_bool == sel_bool

                    compare_func = bool_compare
                else:
                    compare_func = lambda el_val, sel_val: el_val == sel_val
            elif op == "!=":
                compare_func = lambda el_val, sel_val: el_val != sel_val
            elif op == "~=":
                op_desc = f"~= {value!r} (approx)"
                compare_func = lambda el_val, sel_val: _is_approximate_match(el_val, sel_val)
            elif op == "^=":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, str)
                    and isinstance(sel_val, str)
                    and el_val.startswith(sel_val)
                )
            elif op == "$=":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, str)
                    and isinstance(sel_val, str)
                    and el_val.endswith(sel_val)
                )
            elif op == "*=":
                if name == "fontname":
                    op_desc = f"*= {value!r} (contains, case-insensitive)"
                    compare_func = (
                        lambda el_val, sel_val: isinstance(el_val, str)
                        and isinstance(sel_val, str)
                        and sel_val.lower() in el_val.lower()
                    )
                else:
                    op_desc = f"*= {value!r} (contains)"
                    compare_func = (
                        lambda el_val, sel_val: isinstance(el_val, str)
                        and isinstance(sel_val, str)
                        and sel_val in el_val
                    )
            elif op == ">=":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val >= sel_val
                )
            elif op == "<=":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val <= sel_val
                )
            elif op == ">":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val > sel_val
                )
            elif op == "<":
                compare_func = (
                    lambda el_val, sel_val: isinstance(el_val, (int, float))
                    and isinstance(sel_val, (int, float))
                    and el_val < sel_val
                )
            else:
                # Should not happen with current parsing logic
                logger.warning(
                    f"Unsupported operator '{op}' encountered during filter building for attribute '{name}'"
                )
                continue  # Skip this attribute filter

            # --- Create the final filter function for operators with values ---
            filter_name = f"attribute [{name}{op_desc}]"
            # Capture loop variables correctly in the lambda
            filter_lambda = (
                lambda el, get_val=get_element_value, compare=compare_func, expected_val=value: (
                    element_value := get_val(el)
                )
                is not None
                and compare(element_value, expected_val)
            )

        filters.append({"name": filter_name, "func": filter_lambda})

    # Filter by pseudo-classes
    for pseudo in selector["pseudo_classes"]:
        name = pseudo["name"]
        args = pseudo["args"]
        filter_lambda = None
        # Start with a base name, modify for specifics like :not
        filter_name = f"pseudo-class :{name}"

        # Relational pseudo-classes and collection-level pseudo-classes are handled separately by the caller
        if name in ("above", "below", "near", "left-of", "right-of", "first", "last"):
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
            inner_filter_func = selector_to_filter_func(args, aggregates=aggregates, **kwargs)

            # The filter lambda applies the inner function and inverts the result
            filter_lambda = lambda el, inner_func=inner_filter_func: not inner_func(el)

            # Try to create a descriptive name (can be long)
            # Maybe simplify this later if needed
            inner_filter_list = _build_filter_list(args, aggregates=aggregates, **kwargs)
            inner_filter_names = ", ".join([f["name"] for f in inner_filter_list])
            filter_name = f"pseudo-class :not({inner_filter_names})"

        # --- Handle text-based pseudo-classes ---
        elif name == "contains" and args is not None:
            use_regex = kwargs.get("regex", False)
            ignore_case = not kwargs.get("case", True)  # Default case sensitive
            filter_name = (
                f"pseudo-class :contains({args!r}, regex={use_regex}, ignore_case={ignore_case})"
            )

            def contains_check(element, args=args, use_regex=use_regex, ignore_case=ignore_case):
                if not hasattr(element, "text") or not element.text:
                    return False  # Element must have non-empty text

                element_text = element.text
                search_term = str(args)  # Ensure args is string

                if use_regex:
                    try:
                        pattern = re.compile(search_term, re.IGNORECASE if ignore_case else 0)
                        return bool(pattern.search(element_text))
                    except re.error as e:
                        logger.warning(
                            f"Invalid regex '{search_term}' in :contains selector: {e}. Falling back to literal search."
                        )
                        # Fallback to literal search on regex error
                        if ignore_case:
                            return search_term.lower() in element_text.lower()
                        else:
                            return search_term in element_text
                else:  # Literal search
                    if ignore_case:
                        return search_term.lower() in element_text.lower()
                    else:
                        return search_term in element_text

            filter_lambda = contains_check

        # --- Handle :regex pseudo-class (same as :contains with regex=True) ---
        elif name == "regex" and args is not None:
            ignore_case = not kwargs.get("case", True)  # Default case sensitive
            filter_name = f"pseudo-class :regex({args!r}, ignore_case={ignore_case})"

            def regex_check(element, args=args, ignore_case=ignore_case):
                if not hasattr(element, "text") or not element.text:
                    return False  # Element must have non-empty text

                element_text = element.text
                search_term = str(args)  # Ensure args is string

                try:
                    pattern = re.compile(search_term, re.IGNORECASE if ignore_case else 0)
                    return bool(pattern.search(element_text))
                except re.error as e:
                    logger.warning(
                        f"Invalid regex '{search_term}' in :regex selector: {e}. Returning False."
                    )
                    return False

            filter_lambda = regex_check

        # --- Handle :closest pseudo-class for fuzzy text matching --- #
        elif name == "closest" and args is not None:
            # Note: :closest is handled specially in the page._apply_selector method
            # It doesn't filter elements here, but marks them for special processing
            # This allows us to first check :contains matches, then sort by similarity
            filter_lambda = lambda el: True  # Accept all elements for now

        # --- Handle :startswith and :starts-with (alias) --- #
        elif name in ("starts-with", "startswith") and args is not None:
            filter_name = f"pseudo-class :{name}({args!r})"

            def startswith_check(element, arg=args):
                if not hasattr(element, "text") or not element.text:
                    return False
                return str(element.text).startswith(str(arg))

            filter_lambda = startswith_check

        # --- Handle :endswith and :ends-with (alias) --- #
        elif name in ("ends-with", "endswith") and args is not None:
            filter_name = f"pseudo-class :{name}({args!r})"

            def endswith_check(element, arg=args):
                if not hasattr(element, "text") or not element.text:
                    return False
                return str(element.text).endswith(str(arg))

            filter_lambda = endswith_check

        elif name == "starts-with" and args is not None:
            filter_lambda = (
                lambda el, arg=args: hasattr(el, "text")
                and el.text
                and el.text.startswith(str(arg))
            )
        elif name == "ends-with" and args is not None:
            filter_lambda = (
                lambda el, arg=args: hasattr(el, "text") and el.text and el.text.endswith(str(arg))
            )

        # Boolean attribute pseudo-classes
        elif name == "bold":
            filter_lambda = lambda el: hasattr(el, "bold") and el.bold
        elif name == "italic":
            filter_lambda = lambda el: hasattr(el, "italic") and el.italic
        elif name == "horizontal":
            filter_lambda = lambda el: hasattr(el, "is_horizontal") and el.is_horizontal
        elif name == "vertical":
            filter_lambda = lambda el: hasattr(el, "is_vertical") and el.is_vertical
        elif name == "checked":
            filter_lambda = lambda el: hasattr(el, "is_checked") and el.is_checked
        elif name == "unchecked":
            filter_lambda = lambda el: hasattr(el, "is_checked") and not el.is_checked

        # --- New: :strike / :strikethrough / :strikeout pseudo-classes --- #
        elif name in ("strike", "strikethrough", "strikeout"):
            filter_lambda = lambda el: hasattr(el, "strike") and bool(getattr(el, "strike"))
            filter_name = f"pseudo-class :{name}"
        elif name in ("underline", "underlined"):
            filter_lambda = lambda el: hasattr(el, "underline") and bool(getattr(el, "underline"))
            filter_name = f"pseudo-class :{name}"
        elif name in ("highlight", "highlighted"):
            # Match only if the element exposes an `is_highlighted` boolean flag.
            # We deliberately avoid looking at the generic `.highlight()` method on
            # Element, because it is a callable present on every element and would
            # incorrectly mark everything as highlighted.
            filter_lambda = lambda el: bool(getattr(el, "is_highlighted", False))
            filter_name = f"pseudo-class :{name}"

        # Check predefined lambda functions (e.g., :first-child, :empty)
        elif name in PSEUDO_CLASS_FUNCTIONS:
            filter_lambda = PSEUDO_CLASS_FUNCTIONS[name]
            filter_name = f"pseudo-class :{name}"  # Set name for predefined ones
        else:
            raise ValueError(f"Unknown or unsupported pseudo-class: ':{name}'")

        if filter_lambda:
            # Use the potentially updated filter_name
            filters.append({"name": filter_name, "func": filter_lambda})

    return filters


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
            try:
                if not f["func"](element):
                    return False
            except Exception as e:
                logger.error(f"Error applying filter '{f['name']}' to element: {e}", exc_info=True)
                return False  # Treat errors as filter failures
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
    aggregates = {}

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
                try:
                    # Handle special bbox attributes
                    if attr_name in ["x0", "y0", "x1", "y1"]:
                        bbox_mapping = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
                        bbox = getattr(el, "_bbox", None) or getattr(el, "bbox", None)
                        if bbox:
                            val = bbox[bbox_mapping[attr_name]]
                            values.append(val)
                    else:
                        # General attribute access
                        val = getattr(el, attr_name.replace("-", "_"), None)
                        if val is not None:
                            values.append(val)
                except Exception:
                    continue

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


def selector_to_filter_func(
    selector: Dict[str, Any], aggregates: Optional[Dict[str, Any]] = None, **kwargs
) -> Callable[[Any], bool]:
    """
    Convert a parsed selector to a single filter function.

    Internally, this builds a list of individual filters and then combines them.
    To inspect the individual filters, call `_build_filter_list` directly.

    Args:
        selector: Parsed selector dictionary (single or compound OR selector)
        aggregates: Pre-calculated aggregate values (optional)
        **kwargs: Additional filter parameters (e.g., regex, case).

    Returns:
        Function that takes an element and returns True if it matches the selector.
    """
    # Handle compound OR selectors
    if selector.get("type") == "or":
        sub_selectors = selector.get("selectors", [])
        if not sub_selectors:
            # Empty OR selector, return a function that never matches
            return lambda element: False

        # Create filter functions for each sub-selector
        sub_filter_funcs = []
        for sub_selector in sub_selectors:
            sub_filter_funcs.append(
                selector_to_filter_func(sub_selector, aggregates=aggregates, **kwargs)
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Creating OR filter with {len(sub_filter_funcs)} sub-selectors")

        # Return OR combination - element matches if ANY sub-selector matches
        def or_filter(element):
            for func in sub_filter_funcs:
                try:
                    if func(element):
                        return True
                except Exception as e:
                    logger.error(f"Error applying OR sub-filter to element: {e}", exc_info=True)
                    # Continue to next sub-filter on error
                    continue
            return False

        return or_filter

    # Handle single selectors (existing logic)
    filter_list = _build_filter_list(selector, aggregates=aggregates, **kwargs)

    if logger.isEnabledFor(logging.DEBUG):
        filter_names = [f["name"] for f in filter_list]
        logger.debug(f"Assembling filters for selector {selector}: {filter_names}")

    return _assemble_filter_func(filter_list)
