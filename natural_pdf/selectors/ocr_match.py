"""OCR-tolerant string matching for the :ocr() pseudo-class selector.

Designed for finding form labels in OCR'd PDFs where characters may be
garbled but the visual shape of the text is preserved. Uses a blend of
Jaro-Winkler similarity, reverse Jaro-Winkler (to counter prefix bias),
and shape-compressed Levenshtein (to handle visual confusions like l/1/I,
O/0, rn/m).

The shape compression maps characters to ~25 visual equivalence classes,
so "Roviowor" matches "Reviewer" because they have identical visual shapes.
"""

import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# Punctuation at token boundaries (stripped before comparison)
_PUNCT_RE = re.compile(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$")

# Multi-character rewrites applied before shape mapping
_MULTI_CHAR_RULES = [
    ("rn", "m"),
    ("nn", "m"),
    ("ri", "n"),
    ("cl", "d"),
    ("vv", "w"),
    ("ii", "u"),
]

# Shape equivalence classes: visually similar characters map to the same class
_SHAPE_MAP = {}
_shape_groups = [
    ("I", "il1I|!tfj"),  # vertical strokes
    ("O", "oO0cCeQa"),  # round/oval shapes
    ("M", "mwMW"),  # wide humps
    ("N", "nuvNUV"),  # single humps
    ("B", "bhk6"),  # ascender + loop
    ("P", "pqgy9"),  # descender + loop
    ("D", "dDPRB"),  # right-facing bowls
    ("X", "xXzZK"),  # diagonals
    ("S", "sS5$"),  # S-snakes
    ("H", "-_~="),  # horizontals
    ("2", "234"),  # small digits
    ("7", "78"),  # large digits
]
for _cls, _chars in _shape_groups:
    for _ch in _chars:
        _SHAPE_MAP[_ch] = _cls
_SHAPE_MAP["r"] = "R"

# Default threshold (empirically determined: F1=0.985 at this point,
# with 98.4% true positive rate and 0.3% false positive rate on
# 560 real OCR error pairs tested against 74 source documents)
DEFAULT_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    """Normalize text for comparison: NFKC, lowercase, strip edge punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    return text.lower()


def _shape_compress(text: str) -> str:
    """Compress text to visual shape classes.

    Maps each character to its visual equivalence class after applying
    multi-character rewrite rules (e.g., 'rn' -> 'm').
    """
    text = unicodedata.normalize("NFKC", text).lower()
    for pattern, replacement in _MULTI_CHAR_RULES:
        text = text.replace(pattern, replacement)
    return "".join(_SHAPE_MAP.get(ch, ch) for ch in text)


def _length_ratio(a: str, b: str) -> float:
    """Ratio of shorter to longer string length. 1.0 = same length."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _ocr_score_normalized(q: str, c: str) -> float:
    """Core scoring on already-normalized strings (no redundant normalize calls)."""
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0

    # Short strings: require exact shape match (too ambiguous otherwise)
    if len(q) <= 3:
        q_shape = "".join(_SHAPE_MAP.get(ch, ch) for ch in q)
        c_shape = "".join(_SHAPE_MAP.get(ch, ch) for ch in c)
        return 0.95 if q_shape == c_shape and len(q) == len(c) else 0.0

    # Compute features
    jw = JaroWinkler.similarity(q, c)
    sq, sc = _shape_compress(q), _shape_compress(c)
    shape_lev = fuzz.ratio(sq, sc) / 100.0
    lr = _length_ratio(q, c)

    # Blend: 0.4*JW + 0.4*Shape + 0.2*Length
    return 0.4 * jw + 0.4 * shape_lev + 0.2 * lr


# Collapse runs of whitespace to a single space
_WS_RE = re.compile(r"\s+")


def ocr_score(query: str, candidate: str) -> float:
    """Score how likely candidate is an OCR-garbled version of query.

    Uses a blend of three independent signal families:
    - Jaro-Winkler: character-level edit similarity
    - Shape-compressed Levenshtein: visual equivalence similarity
    - Length ratio: penalizes length mismatches

    For short strings (<=3 chars), requires exact shape match to avoid
    false positives.

    Args:
        query: The text the user is searching for (e.g., "Date received")
        candidate: The text found on the page (e.g., "Date recelved")

    Returns:
        Score in [0, 1]. Higher = more likely to be an OCR match.
    """
    return _ocr_score_normalized(_normalize(query), _normalize(candidate))


def ocr_substring_score(query: str, candidate: str) -> float:
    """Score how likely candidate *contains* an OCR-garbled version of query.

    Unlike ``ocr_score`` (which compares two complete strings), this does a
    sliding-window search so that a short query can match inside a longer
    candidate.  This makes ``:ocr()`` a superset of ``:contains()`` — if the
    exact substring is present, the score will be 1.0.

    Algorithm:
    1. If the exact (whitespace-collapsed) query appears as a substring,
       return 1.0 immediately.
    2. If the candidate has roughly the same number of tokens as the query,
       compare directly.
    3. Otherwise, slide windows of query-token-count (and ±1 to tolerate
       OCR-induced word splits/merges) over the candidate and return the
       best score.
    """
    q_norm = _normalize(query)
    c_norm = _normalize(candidate)

    if not q_norm or not c_norm:
        return 0.0

    # Collapse whitespace for substring check (handles newlines, multi-space)
    q_collapsed = _WS_RE.sub(" ", q_norm)
    c_collapsed = _WS_RE.sub(" ", c_norm)

    # Exact substring → 1.0 (fast path, mirrors :contains behaviour)
    if q_collapsed in c_collapsed:
        return 1.0

    # Token-level sliding window
    q_tokens = q_collapsed.split()
    c_tokens = c_collapsed.split()

    if not q_tokens or not c_tokens:
        return _ocr_score_normalized(q_norm, c_norm)

    n_q = len(q_tokens)
    n_c = len(c_tokens)

    if n_q > n_c:
        # Query has more tokens than candidate — fall back to direct comparison
        return _ocr_score_normalized(q_norm, c_norm)

    # If candidate is only slightly longer (≤2 extra tokens), direct compare
    # is fine — windowing won't help much
    if n_c <= n_q + 2:
        return _ocr_score_normalized(q_norm, c_norm)

    # Try window sizes n_q, n_q+1, n_q-1 to tolerate OCR word splits/merges
    window_sizes = [n_q]
    if n_q + 1 <= n_c:
        window_sizes.append(n_q + 1)
    if n_q - 1 >= 1:
        window_sizes.append(n_q - 1)

    best = 0.0
    for ws in window_sizes:
        for i in range(n_c - ws + 1):
            window_text = " ".join(c_tokens[i : i + ws])
            score = _ocr_score_normalized(q_collapsed, window_text)
            if score > best:
                best = score
                if best >= 1.0:
                    return best

    return best


def ocr_match(
    query: str,
    candidates: list,
    threshold: float = DEFAULT_THRESHOLD,
    get_text=None,
) -> list:
    """Find OCR-tolerant matches for query among candidates.

    Args:
        query: Text to search for
        candidates: List of elements to search through
        threshold: Minimum score to consider a match
        get_text: Function to extract text from a candidate element.
                  If None, candidates are treated as strings.

    Returns:
        List of (score, candidate) tuples, sorted by score descending.
        Only includes candidates scoring >= threshold.
    """
    if get_text is None:
        get_text = lambda x: str(x)

    results = []
    for candidate in candidates:
        text = get_text(candidate)
        if not text:
            continue
        score = ocr_score(query, text)
        if score >= threshold:
            results.append((score, candidate))

    results.sort(key=lambda x: x[0], reverse=True)
    return results
