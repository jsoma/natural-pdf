"""Built-in selector clause registrations."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from natural_pdf.selectors.registry import ClauseEvalContext, register_pseudo


@register_pseudo("regex", replace=True)
def _regex_clause(pseudo: Dict[str, Any], ctx: ClauseEvalContext):
    pattern = pseudo.get("args")
    if not isinstance(pattern, str):
        raise ValueError(":regex pseudo-class requires a string argument")

    ignore_case = not ctx.options.get("case", True)
    flags = re.IGNORECASE if ignore_case else 0
    compiled = re.compile(pattern, flags)

    def _filter(element: Any) -> bool:
        text = getattr(element, "text", "")
        if text is None:
            text = ""
        return bool(compiled.search(text))

    return {
        "name": f"pseudo-class :regex({pattern!r})",
        "func": _filter,
    }
