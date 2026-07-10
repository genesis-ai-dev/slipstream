"""constrained_translation.unk_detector — Pre-decode source span detection.

Evidence tier: **Tier 1 (sentence co-occurrence)**.

UNKDetector operates exclusively on the *source* side.  It identifies
contiguous runs of source tokens whose NFKC-lowercased normalised forms do
not appear in the attested target vocabulary (lowercased), and returns them
as UNKSpan records.

### What this module does NOT do

- It does NOT call any model or language backend.
- It does NOT infer source coverage from target subword vocabulary.
- It does NOT generate target text or [UNK:…] markers.
- It does NOT read target strings; it only reads ``source_text`` and the
  pre-built ``attested_vocab`` frozenset that was derived elsewhere.

### Algorithm (§7.5)

1. Whitespace-split ``source_text`` into tokens, recording their character
   offsets in the original string.
2. NFKC-normalise and lowercase each token for comparison.
3. Build a lowercased lookup set from ``attested_vocab``.
4. A token is *covered* if its normalised+lowercased form appears in that
   lookup set.
5. Identify contiguous runs of uncovered tokens.
6. For each run, emit one ``UNKSpan`` whose ``surface`` is the original
   (non-normalised) substring of ``source_text`` spanned by the run, and
   whose ``start_char`` / ``end_char`` are the character offsets of the
   first and last token in the run.

### Invariant I3

``UNKDetector`` is deterministic: given identical inputs it always returns
identical output.  It is called *before* model generation, and its output
is injected into the prompt and grammar independently of generation.  The
model never generates ``[UNK:…]`` text spontaneously.

### Evidence-tier disclaimer (invariant I1)

The ``attested_vocab`` frozenset is derived from target strings of
sentence-co-occurrence pairs (Tier 1).  It does NOT constitute word
alignment.  ``UNKDetector`` uses it only as a *plausibility filter*: if a
source token has no case-insensitively matching surface form in the
attested target vocabulary, we conservatively mark it as unknown.  This is
a deliberate over-approximation — some tokens flagged as unknown may
actually be translatable — but it is the safest Tier-1 policy.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

from constrained_translation.protocol import UNKSpan


class UNKDetector:
    """Identify source spans not lexically represented in the attested vocab.

    Evidence tier: Tier 1 (sentence co-occurrence).  This is NOT word
    alignment.  See module docstring for full disclaimer.
    """

    def detect(
        self,
        source_text: str,
        attested_vocab: frozenset[str],
    ) -> tuple[UNKSpan, ...]:
        """Return UNKSpans for contiguous runs of uncovered source tokens.

        Evidence tier: Tier 1 (sentence co-occurrence).  This is NOT word
        alignment.  Coverage is determined by NFKC-normalised, lowercased
        comparison of source tokens against a lowercased projection of
        ``attested_vocab``.  Surface text in the returned UNKSpans always
        preserves the original characters from ``source_text``.

        Args:
            source_text: The raw source sentence to inspect.  Unicode is
                handled correctly; whitespace splitting is Unicode-aware.
            attested_vocab: A frozenset of surface tokens derived from the
                target sides of the selected example pairs (Tier-1 evidence).

        Returns:
            A tuple of UNKSpan objects in ascending ``start_char`` order.
            Adjacent uncovered tokens are merged into a single span.
            Returns an empty tuple when ``source_text`` has no tokens or
            every token is covered.
        """
        if not source_text or not source_text.strip():
            return ()

        # Build a lowercased lookup set.  NFKC normalisation is applied so
        # that compatibility-equivalent forms (e.g. fullwidth vs. regular)
        # are treated as the same token.
        lookup: set[str] = {
            unicodedata.normalize("NFKC", tok).lower()
            for tok in attested_vocab
        }

        # Tokenise source_text by whitespace, recording character offsets.
        # We iterate manually to capture exact start/end positions.
        tokens: list[tuple[str, int, int]] = []  # (raw_token, start, end)
        i = 0
        n = len(source_text)
        while i < n:
            # Skip whitespace
            while i < n and source_text[i].isspace():
                i += 1
            if i >= n:
                break
            # Consume non-whitespace characters
            j = i
            while j < n and not source_text[j].isspace():
                j += 1
            tokens.append((source_text[i:j], i, j))
            i = j

        if not tokens:
            return ()

        # Classify each token as covered or uncovered.
        covered: list[bool] = []
        for raw_tok, _start, _end in tokens:
            normalised = unicodedata.normalize("NFKC", raw_tok).lower()
            covered.append(normalised in lookup)

        # Merge adjacent uncovered tokens into contiguous runs.
        spans: list[UNKSpan] = []
        run_start: int | None = None  # index into `tokens` list

        for idx, is_covered in enumerate(covered):
            if not is_covered:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    spans.append(_make_span(source_text, tokens, run_start, idx - 1))
                    run_start = None

        # Flush any open run at end of token list
        if run_start is not None:
            spans.append(_make_span(source_text, tokens, run_start, len(tokens) - 1))

        return tuple(spans)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_span(
    source_text: str,
    tokens: list[tuple[str, int, int]],
    first: int,
    last: int,
) -> UNKSpan:
    """Build a UNKSpan from a contiguous run of token indices.

    The ``surface`` is the original substring of ``source_text`` from the
    start of the first token to the end of the last token, preserving all
    original characters (no normalisation).

    Args:
        source_text: The original source string.
        tokens: List of (raw_token, start_char, end_char) tuples.
        first: Index of the first token in the run.
        last: Index of the last token in the run (inclusive).

    Returns:
        A UNKSpan with the correct surface text and character offsets.
    """
    start_char = tokens[first][1]
    end_char = tokens[last][2]
    surface = source_text[start_char:end_char]
    return UNKSpan(surface=surface, start_char=start_char, end_char=end_char)
