"""constrained_translation.vocab_extractor — Attested target-side vocabulary.

Evidence tier: **Tier 1 (sentence co-occurrence)**.  ``evidence_tier=1`` on
every :class:`~constrained_translation.protocol.AlignedExample` records this.

This module derives a frozenset of NFKC-normalised target-language surface
lexical units exclusively from the *target* strings of a collection of
AlignedExample objects.

### What "attested target vocabulary" means at Tier 1

Given a set of selected aligned sentence pairs (source_i, target_i), the
attested vocabulary is the set of whitespace-tokenised surface forms appearing
in any target_i.

**Critical disclaimer (invariant I1):** Sentence co-occurrence is
NOT word alignment.  The fact that a surface form appears in a target sentence
whose source counterpart resembles the query does NOT prove that the form
translates any specific source word.  It merely means the form appears in
translations of sentences related to the query, so we permit the model to use
it as a plausibility filter.

No code or comment in this module may treat Tier-1 evidence as proof of
word-to-word translation correspondence.

### Tokenisation strategy

Tokens are split on Unicode whitespace.  Leading and trailing punctuation is
additionally separated so that, e.g., ``"fin."`` contributes both ``"fin"`` and
``"."`` to the vocabulary.  This ensures punctuation marks are individually
addressable in the grammar builder without requiring the grammar to predict
punctuation attached to words.

Non-ASCII scripts (Greek, Hebrew, Arabic, Devanagari, Ethiopic, etc.) are
handled correctly because the implementation relies on Unicode whitespace
splitting rather than ASCII-only heuristics.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

from constrained_translation.protocol import AlignedExample


# ---------------------------------------------------------------------------
# Helpers for Unicode-aware punctuation splitting.
#
# We split off leading/trailing characters whose Unicode general category is
# in the Punctuation (P*) or Symbol (S*) groups, but we explicitly EXCLUDE
# combining marks (M* categories: Mn, Mc, Me).  Combining marks are
# graphically bonded to the preceding base character and must not be severed —
# this is critical for Arabic diacritics, Devanagari matras, Hebrew vowel
# points, and many other low-resource-language scripts.
# ---------------------------------------------------------------------------

def _is_punct_or_symbol(ch: str) -> bool:
    """Return True iff *ch* is a Unicode punctuation or symbol character.

    Combining marks (category Mn, Mc, Me) return False even though they are
    "non-letter" characters — they must remain attached to their base glyph.
    """
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _split_punctuation(token: str) -> list[str]:
    """Split a single whitespace-delimited token into sub-tokens.

    Leading and trailing *punctuation/symbol* characters are returned as
    separate entries so that ``"fin."`` → ``["fin", "."]`` and
    ``'"cité'`` → ``['"', 'cité']``.

    Combining marks (Arabic diacritics, Devanagari matras, Hebrew niqqud,
    etc.) are never split off because they are integral parts of the
    surrounding grapheme cluster, not stand-alone punctuation.

    Pure-punctuation tokens such as ``"!"`` are returned as a single entry.

    Args:
        token: A single whitespace-free string, already NFKC-normalised.

    Returns:
        A list of 1–3 non-empty strings.
    """
    if not token:
        return []

    # Find the index of the first non-punct/symbol character from the left.
    lead_end = 0
    while lead_end < len(token) and _is_punct_or_symbol(token[lead_end]):
        lead_end += 1

    # Find the index of the last non-punct/symbol character from the right.
    trail_start = len(token)
    while trail_start > lead_end and _is_punct_or_symbol(token[trail_start - 1]):
        trail_start -= 1

    lead = token[:lead_end]
    body = token[lead_end:trail_start]
    trail = token[trail_start:]

    parts: list[str] = []
    if lead:
        parts.append(lead)
    if body:
        parts.append(body)
    if trail:
        parts.append(trail)
    return parts if parts else [token]


class VocabExtractor:
    """Derive an attested target-side vocabulary from AlignedExample objects.

    Evidence tier: Tier 1 (sentence co-occurrence).  See module docstring for
    the critical non-alignment disclaimer.
    """

    def extract(self, examples: Iterable[AlignedExample]) -> frozenset[str]:
        """Return the union of NFKC-normalised surface tokens from all target strings.

        Evidence tier: Tier 1 (sentence co-occurrence), recorded as
        ``evidence_tier=1`` on each AlignedExample.  This is NOT word alignment.
        The returned tokens are surface forms that appear in target
        translations of sentences related to the query; they do not constitute
        evidence that any individual token translates a specific source word.

        Algorithm:
          1. Iterate over *target* strings only — source strings are never read.
          2. Split each target string on Unicode whitespace.
          3. Apply NFKC normalisation to each whitespace-delimited token.
          4. Separate leading/trailing punctuation into individual entries.
          5. Return the union of all non-empty tokens as a frozenset.

        Args:
            examples: Any iterable of AlignedExample objects.  Generators are
                accepted.  An empty iterable returns an empty frozenset.

        Returns:
            frozenset[str] of NFKC-normalised surface tokens derived exclusively
            from the *target* strings of the supplied examples.
        """
        tokens: set[str] = set()

        for example in examples:
            # Read ONLY the target string — never example.source.
            raw_target: str = example.target
            if not raw_target:
                continue

            # Split on Unicode whitespace.
            for raw_tok in raw_target.split():
                if not raw_tok:
                    continue
                # NFKC-normalise the raw whitespace token.
                nfkc_tok = unicodedata.normalize("NFKC", raw_tok)
                # Separate leading/trailing punctuation.
                for sub in _split_punctuation(nfkc_tok):
                    if sub:
                        tokens.add(sub)

        return frozenset(tokens)
