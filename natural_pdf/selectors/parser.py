"""
CSS-like selector parser for natural-pdf.
"""

import ast
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from colour import Color

logger = logging.getLogger(__name__)


def safe_parse_value(value_str: str) -> Any:
    """
    Safely parse a value string without using eval().

    Args:
        value_str: String representation of a value (number, tuple, string, etc.)

    Returns:
        Parsed value
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


def safe_parse_color(value_str: str) -> tuple:
    """
    Parse a color value which could be an RGB tuple, color name, or hex code.

    Args:
        value_str: String representation of a color (e.g., "red", "#ff0000", "(1,0,0)")

    Returns:
        RGB tuple (r, g, b) with values from 0 to 1
    """
    value_str = value_str.strip()

    # Try parsing as a Python literal (for RGB tuples)
    try:
        # If it's already a valid tuple or list, parse it
        color_tuple = ast.literal_eval(value_str)
        if isinstance(color_tuple, (list, tuple)) and len(color_tuple) >= 3:
            # Return just the RGB components as a tuple
            return tuple(color_tuple[:3])
    except (SyntaxError, ValueError):
        # Not a valid tuple/list, try as a color name or hex
        try:
            # Use colour library to parse color names, hex values, etc.
            color = Color(value_str)
            # Convert to RGB tuple with values between 0 and 1
            return (color.red, color.green, color.blue)
        except (ValueError, AttributeError):
            # If color parsing fails, return a default (black)
            return (0, 0, 0)

    # If we got here with a non-tuple, return default
    return (0, 0, 0)


def parse_selector(selector: str) -> Dict[str, Any]:
    """
    Parse a CSS-like selector string into a structured selector object.

    Handles:
    - Element types (e.g., 'text', 'rect')
    - Attribute presence (e.g., '[data-id]')
    - Attribute value checks with various operators (e.g., '[count=5]', '[name*="bold"]'')
    - Pseudo-classes (e.g., ':contains("Total")', ':empty', ':not(...)')

    Args:
        selector: CSS-like selector string

    Returns:
        Dict representing the parsed selector
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

    # --- Handle wildcard selector explicitly ---
    if selector == "*":
        # Wildcard matches any type, already the default.
        # Clear selector so the loop doesn't run and error out.
        selector = ""
    # --- END NEW ---

    # 1. Extract type (optional, at the beginning)
    # Only run if selector wasn't '*'
    if selector:
        type_match = re.match(r"^([a-zA-Z_\-]+)", selector)
        if type_match:
            result["type"] = type_match.group(1).lower()
            selector = selector[len(type_match.group(0)) :].strip()
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
                # Parse value
                parsed_value: Any
                if name in [
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
            name, args_str = pseudo_match.groups()
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
            selector = selector[pseudo_match.end() :].strip()
            processed_chunk = True
            continue

        # If we reach here and the selector string is not empty, something is wrong
        if not processed_chunk and selector:
            raise ValueError(
                f"Invalid or unexpected syntax near '{selector[:30]}...'. Full selector: '{original_selector_for_error}'"
            )

    return result


def _is_approximate_match(value1, value2, tolerance: float = 0.1) -> bool:
    """
    Check if two values approximately match.

    This is mainly used for color comparisons with some tolerance.

    Args:
        value1: First value
        value2: Second value
        tolerance: Maximum difference allowed

    Returns:
        True if the values approximately match
    """
    # Handle string colors by converting them to RGB tuples
    if isinstance(value1, str):
        try:
            value1 = tuple(Color(value1).rgb)
        except:
            pass

    if isinstance(value2, str):
        try:
            value2 = tuple(Color(value2).rgb)
        except:
            pass

    # If both are tuples/lists with the same length (e.g., colors)
    if (
        isinstance(value1, (list, tuple))
        and isinstance(value2, (list, tuple))
        and len(value1) == len(value2)
    ):

        # Check if all components are within tolerance
        return all(abs(a - b) <= tolerance for a, b in zip(value1, value2))

    # If both are numbers
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        return abs(value1 - value2) <= tolerance

    # Default to exact match for other types
    return value1 == value2


PSEUDO_CLASS_FUNCTIONS = {
    "bold": lambda el: hasattr(el, "bold") and el.bold,
    "italic": lambda el: hasattr(el, "italic") and el.italic,
    "first-child": lambda el: hasattr(el, "parent") and el.parent and el.parent.children[0] == el,
    "last-child": lambda el: hasattr(el, "parent") and el.parent and el.parent.children[-1] == el,
    "empty": lambda el: not el.text,
    "not-empty": lambda el: el.text,
    "not-bold": lambda el: hasattr(el, "bold") and not el.bold,
    "not-italic": lambda el: hasattr(el, "italic") and not el.italic,
}


def _build_filter_list(selector: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    Convert a parsed selector to a list of named filter functions.

    Args:
        selector: Parsed selector dictionary
        **kwargs: Additional filter parameters including:
                 - regex: Whether to use regex for text search
                 - case: Whether to do case-sensitive text search

    Returns:
        List of dictionaries, each with 'name' (str) and 'func' (callable).
        The callable takes an element and returns True if it matches the specific filter.
    """
    filters: List[Dict[str, Any]] = []
    selector_type = selector["type"]

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
                compare_func = lambda el_val, sel_val: el_val == sel_val
            elif op == "!=":
                compare_func = lambda el_val, sel_val: el_val != sel_val
            elif op == "~":
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

        # Relational pseudo-classes are handled separately by the caller
        if name in ("above", "below", "near", "left-of", "right-of"):
            continue

        # --- Handle :not() ---
        elif name == "not":
            if not isinstance(args, dict):  # args should be the parsed inner selector
                logger.error(f"Invalid arguments for :not pseudo-class: {args}")
                raise TypeError(
                    "Internal error: :not pseudo-class requires a parsed selector dictionary as args."
                )

            # Recursively get the filter function for the inner selector
            # Pass kwargs down in case regex/case flags affect the inner selector
            inner_filter_func = selector_to_filter_func(args, **kwargs)

            # The filter lambda applies the inner function and inverts the result
            filter_lambda = lambda el, inner_func=inner_filter_func: not inner_func(el)

            # Try to create a descriptive name (can be long)
            # Maybe simplify this later if needed
            inner_filter_list = _build_filter_list(args, **kwargs)
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


def selector_to_filter_func(selector: Dict[str, Any], **kwargs) -> Callable[[Any], bool]:
    """
    Convert a parsed selector to a single filter function.

    Internally, this builds a list of individual filters and then combines them.
    To inspect the individual filters, call `_build_filter_list` directly.

    Args:
        selector: Parsed selector dictionary
        **kwargs: Additional filter parameters (e.g., regex, case).

    Returns:
        Function that takes an element and returns True if it matches the selector.
    """
    filter_list = _build_filter_list(selector, **kwargs)

    if logger.isEnabledFor(logging.DEBUG):
        filter_names = [f["name"] for f in filter_list]
        logger.debug(f"Assembling filters for selector {selector}: {filter_names}")

    return _assemble_filter_func(filter_list)
