from typing import List, Tuple

import numpy as np


def group_cells_into_rows_and_columns(
    cell_boxes: List[Tuple[float, float, float, float]],
    row_tol: float = None,
    col_tol: float = None,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """
    Groups cell bounding boxes into rows and columns using spatial proximity.

    Args:
        cell_boxes: List of (x0, y0, x1, y1) for each cell.
        row_tol: Vertical tolerance for grouping rows (default: 10% of median cell height).
        col_tol: Horizontal tolerance for grouping columns (default: 10% of median cell width).

    Returns:
        (row_boxes, col_boxes): Lists of bounding boxes for rows and columns.
    """
    if not cell_boxes:
        return [], []

    # Convert to numpy for easier manipulation
    boxes = np.array(cell_boxes)
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]

    # Set default tolerances if not provided
    median_height = float(np.median(heights))
    median_width = float(np.median(widths))
    row_tol = row_tol if row_tol is not None else max(2.0, 0.1 * median_height)
    col_tol = col_tol if col_tol is not None else max(2.0, 0.1 * median_width)

    # --- Group into rows ---
    row_groups = []
    for i, yc in enumerate(y_centers):
        placed = False
        for group in row_groups:
            # If this cell's center is close to the group's mean center, add it
            if abs(yc - np.mean([y_centers[j] for j in group])) <= row_tol:
                group.append(i)
                placed = True
                break
        if not placed:
            row_groups.append([i])

    # --- Group into columns ---
    col_groups = []
    for i, xc in enumerate(x_centers):
        placed = False
        for group in col_groups:
            if abs(xc - np.mean([x_centers[j] for j in group])) <= col_tol:
                group.append(i)
                placed = True
                break
        if not placed:
            col_groups.append([i])

    # --- Compute bounding boxes for each group ---
    row_boxes = []
    for group in row_groups:
        x0 = float(np.min(boxes[group, 0]))
        y0 = float(np.min(boxes[group, 1]))
        x1 = float(np.max(boxes[group, 2]))
        y1 = float(np.max(boxes[group, 3]))
        row_boxes.append((x0, y0, x1, y1))

    col_boxes = []
    for group in col_groups:
        x0 = float(np.min(boxes[group, 0]))
        y0 = float(np.min(boxes[group, 1]))
        x1 = float(np.max(boxes[group, 2]))
        y1 = float(np.max(boxes[group, 3]))
        col_boxes.append((x0, y0, x1, y1))

    return row_boxes, col_boxes
