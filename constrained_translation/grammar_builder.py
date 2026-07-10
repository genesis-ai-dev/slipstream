"""constrained_translation.grammar_builder — Per-request XGrammar EBNF grammar.

Builds a grammar string that licenses *only* attested NFKC target surface units
plus an explicit minimal punctuation set and a canonical single-space separator.

## Grammar design

The produced grammar is valid EBNF as consumed by XGrammar.  Its structure
prevents unattested lexical concatenation (e.g. ``terreAu`` from adjacent
tokens ``terre`` + ``Au``) by requiring an explicit space separator between
every pair of lexical units:

    root           ::= translation
    translation    ::= (" ")* lex_core (" ")* "\\n"?
    lex_core       ::= tok_unit (" " tok_unit)*
    tok_unit       ::= PUNCT* attested_token PUNCT*
    PUNCT          ::= "." | "," | "!" | "?" | ";" | ":" | "'" | "\\""
    attested_token ::= "word1" | "word2" | ...

Key structural invariants
--------------------------
* Two attested surface units can **never** appear adjacent without an
  intervening ``" "`` — the separator is mandatory in ``lex_core``.
* At least one ``attested_token`` is always required (``lex_core`` →
  ``tok_unit`` → ``attested_token``).  Whitespace/punctuation-only output
  is structurally impossible.
* Canonical single space as inter-token separator; optional leading/trailing
  spaces and a single optional final newline.  No unbounded arbitrary
  whitespace sequences between tokens.
* Punctuation may be freely attached before and after any lexical unit
  (``PUNCT*`` on both sides of ``attested_token``).

Key invariants inherited from §7.6
-------------------------------------
* I2 — grammar is over surface tokens, not subword IDs.
* I3 — ``[UNK:…]`` markers are **never** included; they are inserted
  deterministically by ``UNKDetector`` after generation.
* I7 — if ``attested_vocab`` is empty, ``GrammarBuildError`` is raised
  immediately.  No silent fallback.

## Escaping

EBNF string literals are enclosed in double-quotes.  The following characters
are escaped inside those literals:

* ``"``  → ``\\"``   — closes the surrounding double-quote literal.
* ``\\`` → ``\\\\``   — backslash must be doubled first to avoid
  double-processing.

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
      (attached directly to tokens with no required separator).
    - Optional leading/trailing spaces and a single optional final newline.
    - Exactly one canonical single space as the mandatory separator between
      adjacent lexical units (prevents unattested concatenation).

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

        The grammar enforces:
        - At least one attested lexical unit (empty / whitespace-only outputs
          are rejected structurally).
        - A mandatory single-space separator between adjacent lexical units
          (prevents unattested concatenations such as ``terreAu``).
        - Optional leading/trailing whitespace (one or more spaces) and a
          single optional trailing newline.
        - Punctuation may appear freely attached to either side of any
          attested token — no separator required between punctuation and token.

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

        # Reject tokens containing Unicode control/surrogate/private-use/unassigned
        # characters (categories Cc, Cf, Cs, Co, Cn).  These characters have no
        # legitimate place inside a surface token and would silently corrupt the
        # grammar or the generated output.
        #
        # Narrow allowlist for Cf: zero-width joining/spacing characters that are
        # linguistically essential in real corpora and are NOT security-sensitive:
        #   U+200B  ZERO WIDTH SPACE          — Burmese (mya) word-boundary marker
        #   U+200C  ZERO WIDTH NON-JOINER     — Devanagari / Indic scripts
        #   U+200D  ZERO WIDTH JOINER         — Nepali (npi) conjunct consonants
        # All other Cf characters (bidi overrides, soft-hyphen, language tags, …)
        # remain forbidden.
        _CF_ALLOWLIST: frozenset[str] = frozenset({
            "\u200B",  # ZERO WIDTH SPACE
            "\u200C",  # ZERO WIDTH NON-JOINER
            "\u200D",  # ZERO WIDTH JOINER
        })
        _FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})
        for tok in attested_vocab:
            for ch in tok:
                cat = unicodedata.category(ch)
                if cat in _FORBIDDEN_CATEGORIES and ch not in _CF_ALLOWLIST:
                    raise GrammarBuildError(
                        item_id=item_id,
                        reason=(
                            f"attested token {tok!r} contains a control/non-graphic "
                            f"character U+{ord(ch):04X} (Unicode category {cat})"
                        ),
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
        #
        # Structural design to prevent concatenation without separator:
        #
        #   translation  ::= (" ")* lex_core (" ")* "\n"?
        #   lex_core     ::= tok_unit (" " tok_unit)*
        #   tok_unit     ::= PUNCT* attested_token PUNCT*
        #
        # This means:
        # - Between any two attested_token runs exactly one " " is required.
        # - Leading/trailing spaces are optional.
        # - A single final "\n" is optional.
        # - Punctuation may be attached freely on either side of a token.
        # - At least one attested_token is required (lex_core is not optional).
        lines: list[str] = [
            'root           ::= translation',
            'translation    ::= (" ")* lex_core (" ")* "\\n"?',
            'lex_core       ::= tok_unit (" " tok_unit)*',
            'tok_unit       ::= PUNCT* attested_token PUNCT*',
            'PUNCT          ::= ' + ' | '.join(punct_alts),
            'attested_token ::= ' + ' | '.join(token_alts),
        ]

        return "\n".join(lines) + "\n"
