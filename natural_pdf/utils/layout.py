from typing import List, Optional, Tuple


def merge_bboxes(
    bboxes: List[Optional[Tuple[float, float, float, float]]],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Merge multiple bounding boxes into a single one that encompasses all of them.

    Args:
        bboxes: A list of bbox tuples (x0, top, x1, bottom). Can contain None values.

    Returns:
        A single merged bbox tuple, or None if no valid bboxes are provided.
    """
    if not bboxes:
        return None

    # Filter out None or invalid bboxes
    valid_bboxes = [b for b in bboxes if b and len(b) == 4]
    if not valid_bboxes:
        return None

    x0s, tops, x1s, bottoms = zip(*valid_bboxes)

    return (min(x0s), min(tops), max(x1s), max(bottoms))
