"""constrained_translation.grammar_builder — Per-request XGrammar EBNF grammar.

Builds a grammar string that licenses *only* attested NFKC target surface units
plus an explicit minimal punctuation set and whitespace/newline.

## Grammar design

The produced grammar is valid EBNF as consumed by XGrammar.  Its structure
mirrors the plan (§7.6):

    root         ::= translation
    translation  ::= segment+
    segment      ::= attested_token | punctuation | WS
    attested_token ::= "word1" | "word2" | ...
    punctuation  ::= "." | "," | "!" | "?" | ";" | ":" | "'" | "\""
    WS           ::= (" " | "\\n")+

Key invariants enforced here
-----------------------------
* I2 — grammar is over surface tokens, not subword IDs.
* I3 — ``[UNK:…]`` markers are **never** included; they are inserted
  deterministically by ``UNKDetector`` after generation.
* I7 — if ``attested_vocab`` is empty, ``GrammarBuildError`` is raised
  immediately.  No silent fallback.

## Escaping

EBNF string literals are enclosed in double-quotes.  The following characters
are escaped inside those literals:

* ``"``  → ``\\"``
* ``\\`` → ``\\\\``
* ``\\n`` → ``\\\\n`` (to insert a literal backslash-n, not a newline)

All other characters — including Unicode letters, digits, apostrophes,
brackets, and parentheses — are passed through verbatim.  XGrammar treats
everything inside double-quoted literals as a literal byte/character sequence,
so regex meta-characters like ``+``, ``*``, ``?``, ``[``, ``]`` are safe.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class GrammarBuildError(Exception):
    """Raised when a grammar cannot be built (invariant I7).

    Attributes:
        item_id: The translation-request identifier for which building failed.
        reason:  Human-readable explanation of the failure.
    """

    def __init__(self, item_id: str, reason: str) -> None:
        super().__init__(f"[{item_id}] {reason}")
        self.item_id = item_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Minimal punctuation set (plan §7.6)
# ---------------------------------------------------------------------------

_PUNCTUATION_CHARS: tuple[str, ...] = (
    ".", ",", "!", "?", ";", ":", "'", '"',
)


# ---------------------------------------------------------------------------
# Escaping helpers
# ---------------------------------------------------------------------------

def _escape_ebnf_literal(token: str) -> str:
    """Escape *token* for safe use inside a double-quoted EBNF string literal.

    Rules applied (in order):
    1. ``\\`` → ``\\\\``   — backslash must be doubled first to avoid
       double-processing.
    2. ``"``  → ``\\"``   — closes the surrounding double-quote literal.

    All other characters — Unicode letters, digits, punctuation characters
    other than the two above, combining marks — are passed through verbatim.
    XGrammar treats everything inside double-quoted literals as a literal
    character sequence, so no further escaping is required.

    Args:
        token: An NFKC-normalised surface string.

    Returns:
        The escaped string ready to be embedded as ``"<escaped>"`` in EBNF.
    """
    # Step 1: escape backslashes first (must precede all other escaping).
    token = token.replace("\\", "\\\\")
    # Step 2: escape double-quotes.
    token = token.replace('"', '\\"')
    return token


# ---------------------------------------------------------------------------
# GrammarBuilder
# ---------------------------------------------------------------------------

class GrammarBuilder:
    """Build a per-request XGrammar-compatible EBNF grammar string.

    The grammar licenses *only*:
    - Attested NFKC target surface tokens supplied in ``attested_vocab``.
    - A fixed minimal punctuation set: . , ! ? ; : ' "
    - Whitespace (spaces) and a sentence-final newline.

    [UNK:…] markers are NEVER licensed — they are inserted deterministically
    by ``UNKDetector`` outside model generation (invariant I3).

    Raises:
        GrammarBuildError: If ``attested_vocab`` is empty (invariant I7).
    """

    def build(
        self,
        attested_vocab: frozenset[str],
        item_id: str,
    ) -> str:
        """Build and return the EBNF grammar string.

        Args:
            attested_vocab: NFKC-normalised surface tokens from the target
                sides of the retrieved examples (Tier-1 evidence).  Must be
                non-empty.
            item_id: Identifier for the translation request; stored in any
                raised ``GrammarBuildError``.

        Returns:
            A multi-line EBNF grammar string compatible with XGrammar.

        Raises:
            GrammarBuildError: If *attested_vocab* is empty.
        """
        if not attested_vocab:
            raise GrammarBuildError(
                item_id=item_id,
                reason="attested_vocab is empty — cannot build a grammar with no tokens",
            )

        # Build quoted alternatives for each attested surface token.
        # Sort for determinism.
        token_alts: list[str] = [
            f'"{_escape_ebnf_literal(tok)}"'
            for tok in sorted(attested_vocab)
        ]

        # Build quoted alternatives for the minimal punctuation set.
        punct_alts: list[str] = [
            f'"{_escape_ebnf_literal(ch)}"'
            for ch in _PUNCTUATION_CHARS
        ]

        # Assemble grammar rules.
        lines: list[str] = [
            "root         ::= translation",
            "translation  ::= segment+",
            "segment      ::= attested_token | punctuation | WS",
            "attested_token ::= " + " | ".join(token_alts),
            "punctuation  ::= " + " | ".join(punct_alts),
            r'WS           ::= (" " | "\n")+',
        ]

        return "\n".join(lines) + "\n"
