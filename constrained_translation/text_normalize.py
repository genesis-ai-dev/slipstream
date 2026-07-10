"""constrained_translation.text_normalize — Shared source-unit normalization.

Single authoritative helper for breaking raw source text into normalised
lexical units used consistently across:

  * ExampleSelector  — coverage set computation (query units & doc units)
  * BatchRunner      — building the source-derived vocabulary for UNKDetector
  * UNKDetector      — normalizing source tokens for lookup comparison

Algorithm
─────────
1. NFKC-normalise   → collapses Unicode compatibility variants
2. Lowercase        → case-insensitive comparison
3. Strip punctuation → removes any character that is not a word char or
                       whitespace (``[^\\w\\s]`` regex, Unicode-aware)
4. Collapse whitespace → single spaces between tokens
5. Split on whitespace → return list of non-empty token strings

This is deliberately *compatible* with the existing ``BM25Query._normalize_text``
behaviour on ASCII English text (that function does steps 2-4 but not 1).
NFKC is added here for correct Unicode handling; it is a no-op on pure ASCII.

Punctuation-only tokens
───────────────────────
A raw whitespace token that consists entirely of punctuation (e.g. ``","``
or ``"..."`` ) normalises to the empty string after step 3 and is **not**
returned in the output list.  Callers that care about the original token
boundaries (UNKDetector) must treat such tokens as covered / transparent
so they never become spurious [UNK] spans.
"""

from __future__ import annotations

import re
import unicodedata

# Pre-compiled regex for punctuation removal (same pattern as BM25Query._normalize_text)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_source_units(text: str) -> list[str]:
    """Return the list of normalised lexical units for *text*.

    Steps: NFKC  →  lowercase  →  strip punctuation  →  collapse whitespace
    →  split.  Empty results (punctuation-only input) produce an empty list.

    Parameters
    ----------
    text:
        Raw source text (a sentence, a single token, or an empty string).

    Returns
    -------
    list[str]
        Non-empty normalised tokens in order.  Returns ``[]`` for empty or
        punctuation-only inputs.

    Examples
    --------
    >>> normalize_source_units("In the beginning,")
    ['in', 'the', 'beginning']
    >>> normalize_source_units("you,,")
    ['you']
    >>> normalize_source_units(",")
    []
    >>> normalize_source_units("don't stop")
    ['dont', 'stop']
    >>> normalize_source_units("well-known")
    ['wellknown']
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.split() if text else []
