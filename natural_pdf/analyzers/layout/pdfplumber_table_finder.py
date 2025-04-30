from typing import Any, Dict, List, Optional, Tuple, Union

# Attempt to import pdfplumber modules
import pdfplumber.table as pdfplumber_table

# Type Definitions
T_num = Union[int, float]
T_bbox = Tuple[T_num, T_num, T_num, T_num]
T_obj = Dict[str, Any]
T_obj_list = List[T_obj]
T_intersections = Dict[Tuple[T_num, T_num], Dict[str, T_obj_list]]
T_cell_dict = Dict[str, T_num]

# Use defaults directly from pdfplumber.table (or placeholders if not installed)
DEFAULT_SNAP_TOLERANCE = pdfplumber_table.DEFAULT_SNAP_TOLERANCE
DEFAULT_JOIN_TOLERANCE = pdfplumber_table.DEFAULT_JOIN_TOLERANCE
DEFAULT_MIN_WORDS_VERTICAL = pdfplumber_table.DEFAULT_MIN_WORDS_VERTICAL
DEFAULT_MIN_WORDS_HORIZONTAL = pdfplumber_table.DEFAULT_MIN_WORDS_HORIZONTAL

# --- Main Function ---


def find_text_based_tables(
    bboxes: List[T_bbox],
    snap_tolerance: T_num = DEFAULT_SNAP_TOLERANCE,
    join_tolerance: T_num = DEFAULT_JOIN_TOLERANCE,
    min_words_vertical: int = DEFAULT_MIN_WORDS_VERTICAL,
    min_words_horizontal: int = DEFAULT_MIN_WORDS_HORIZONTAL,
    intersection_tolerance: T_num = 3,
    snap_x_tolerance: Optional[T_num] = None,
    snap_y_tolerance: Optional[T_num] = None,
    join_x_tolerance: Optional[T_num] = None,
    join_y_tolerance: Optional[T_num] = None,
    intersection_x_tolerance: Optional[T_num] = None,
    intersection_y_tolerance: Optional[T_num] = None,
) -> Dict[str, Union[T_obj_list, List[T_cell_dict], T_intersections]]:
    """
    Finds table structures based on text element alignment using imported
    pdfplumber functions. Accepts a list of bounding box tuples.

    Args:
        bboxes: A list of bounding box tuples (x0, top, x1, bottom).
        snap_tolerance: General tolerance for snapping edges.
        join_tolerance: General tolerance for joining nearby edges.
        min_words_vertical: Minimum words to form a vertical edge.
        min_words_horizontal: Minimum words to form a horizontal edge.
        intersection_tolerance: General tolerance for intersections.
        snap_x_tolerance: Specific horizontal snap tolerance (overrides general).
        snap_y_tolerance: Specific vertical snap tolerance (overrides general).
        join_x_tolerance: Specific horizontal join tolerance (overrides general).
        join_y_tolerance: Specific vertical join tolerance (overrides general).
        intersection_x_tolerance: Specific horizontal intersection tolerance.
        intersection_y_tolerance: Specific vertical intersection tolerance.


    Returns:
        A dictionary containing:
        - 'horizontal_edges': List of merged horizontal edge dictionaries.
        - 'vertical_edges': List of merged vertical edge dictionaries.
        - 'cells': List of dictionaries [{'left': x0, 'top': top, 'right': x1, 'bottom': bottom}, ...]
                   representing detected cells, ready for page.region().
        - 'intersections': Dictionary of intersection points and the edges forming them.

    Raises:
        ImportError: If the 'pdfplumber' library is not installed when this function is called.
    """

    if not bboxes:
        return {"horizontal_edges": [], "vertical_edges": [], "cells": [], "intersections": {}}

    # Convert BBoxes to Dictionaries required by pdfplumber functions
    text_elements = []
    for i, bbox in enumerate(bboxes):
        x0, top, x1, bottom = bbox
        # Basic structure needed for words_to_edges_h/v
        text_elements.append(
            {
                "x0": x0,
                "top": top,
                "x1": x1,
                "bottom": bottom,
                "width": x1 - x0,
                "height": bottom - top,
                "text": f"elem_{i}",  # Placeholder text
                "object_type": "char",  # Mimic word/char structure loosely
            }
        )

    # Resolve tolerances
    sx = snap_x_tolerance if snap_x_tolerance is not None else snap_tolerance
    sy = snap_y_tolerance if snap_y_tolerance is not None else snap_tolerance
    jx = join_x_tolerance if join_x_tolerance is not None else join_tolerance
    jy = join_y_tolerance if join_y_tolerance is not None else join_tolerance
    ix = (
        intersection_x_tolerance if intersection_x_tolerance is not None else intersection_tolerance
    )
    iy = (
        intersection_y_tolerance if intersection_y_tolerance is not None else intersection_tolerance
    )

    # --- pdfplumber Pipeline ---
    h_edges = pdfplumber_table.words_to_edges_h(text_elements, word_threshold=min_words_horizontal)
    v_edges = pdfplumber_table.words_to_edges_v(text_elements, word_threshold=min_words_vertical)
    initial_edges = h_edges + v_edges

    if not initial_edges:
        return {"horizontal_edges": [], "vertical_edges": [], "cells": [], "intersections": {}}

    merged_edges = pdfplumber_table.merge_edges(initial_edges, sx, sy, jx, jy)
    merged_h = [e for e in merged_edges if e["orientation"] == "h"]
    merged_v = [e for e in merged_edges if e["orientation"] == "v"]

    if not merged_edges:
        return {
            "horizontal_edges": merged_h,
            "vertical_edges": merged_v,
            "cells": [],
            "intersections": {},
        }

    intersections = pdfplumber_table.edges_to_intersections(merged_edges, ix, iy)
    if not intersections:
        return {
            "horizontal_edges": merged_h,
            "vertical_edges": merged_v,
            "cells": [],
            "intersections": intersections,
        }

    cell_tuples = pdfplumber_table.intersections_to_cells(intersections)

    # Convert cell tuples to dictionaries for page.region()
    cell_dicts = []
    for x0, top, x1, bottom in cell_tuples:
        cell_dicts.append({"left": x0, "top": top, "right": x1, "bottom": bottom})

    return {
        "horizontal_edges": merged_h,
        "vertical_edges": merged_v,
        "cells": cell_dicts,
        "intersections": intersections,
    }
