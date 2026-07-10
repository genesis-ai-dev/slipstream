"""constrained_translation.token_auditor — Exact token-ID and surface-composition
provenance audit (Task 8, invariant I2).

Overview
--------
The TokenAuditor verifies that every token ID produced by the generation
backend can be *traced back* to a selected few-shot target example.  It
enforces two complementary invariants:

1. **ID-level licence** — every generated token ID must appear in the set of
   IDs that the tokenizer assigns to at least one complete example ``target``
   string (the *token licence*).

2. **Surface-composition check** — even if every subword piece ID is licenced,
   the licensed tokens are *reconstructed* into full surface text using the
   backend's ``decode_tokens()`` (full-sequence detokenisation), and each
   resulting surface word must appear in ``attested_vocab``.  This prevents
   the model from recombining licenced subword pieces into novel, unattested
   words.

   ``decode_tokens()`` is used (rather than concatenating individual
   ``decode_token()`` results) because BPE tokenisers such as Qwen3.5 and
   byte-level tokenisers can produce garbled output when tokens are decoded
   individually:

   * **BPE partial-word splits** — "créa" may tokenise as [cré_id, a_id].
     ``decode_token(cré_id)`` → "cré"; ``decode_token(a_id)`` → "a".
     Neither "cré" nor "a" is in ``attested_vocab``, causing a false
     violation.  ``decode_tokens([cré_id, a_id])`` → "créa" (correct).

   * **Byte-level BPE fragments** — a single Burmese Unicode character
     (UTF-8: 3 bytes) may split into three byte tokens, each decoding to
     U+FFFD in isolation.  ``decode_tokens`` of the full sequence returns
     the actual character.

3. **UNK-marker rejection** — tokens whose decoded surface matches the pattern
   ``[UNK:…]`` always fail, regardless of licence status (invariant I3: UNK
   markers are inserted deterministically by ``UNKDetector`` before generation;
   the model must never produce them spontaneously).

4. **Layout narrow rule** — tokens that decode to pure whitespace or pure
   Unicode punctuation/symbol characters are silently passed as *layout*
   tokens.  They carry no lexical provenance and do not appear in the returned
   provenance map.

API
---
.. code-block:: python

    auditor = TokenAuditor()

    # Build a token-ID licence from examples (calls backend.tokenize per target).
    licence: TokenLicense = auditor.build_license(examples, backend)

    # Audit generated token IDs.
    result, provenance = auditor.audit(
        token_ids=gen_result.token_ids,
        attested_vocab=attested_vocab,
        backend=backend,
        examples=examples,          # builds licence on the fly, OR
        license=precomputed_licence,  # use a precomputed one
    )

``TokenLicense``
    ``dict[int, list[int]]`` — maps each licensed token ID to a list of
    ``verse_idx`` values identifying which examples contributed that ID.

Provenance map (second return value of ``audit``)
    A ``TokenLicense`` containing only the IDs that were actually *seen* during
    this audit call and passed as non-layout tokens.  It is a convenient
    sub-slice of the full licence, usable by downstream loggers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

from constrained_translation.protocol import AlignedExample, BackendProtocol, TokenAuditResult

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Maps token_id → list[verse_idx] (provenance)
TokenLicense = dict[int, list[int]]

# Matches model-generated UNK markers, e.g. "[UNK:selah]"
_UNK_MARKER_RE = re.compile(r"\[UNK:[^\]]*\]")

# Word-boundary marker used by SentencePiece (U+2581 LOWER ONE EIGHTH BLOCK)
_WORD_BOUNDARY = "\u2581"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_layout(decoded: str) -> bool:
    """Return True if *decoded* consists entirely of whitespace and/or Unicode
    punctuation/symbol characters (the layout narrow rule).

    An empty string is treated as layout (nothing to audit).
    """
    # Normalise the ▁ word-boundary marker away before checking.
    surface = decoded.replace(_WORD_BOUNDARY, "")
    if not surface:
        return True  # pure boundary marker or empty → layout
    for ch in surface:
        cat = unicodedata.category(ch)
        if not (ch.isspace() or cat.startswith("P") or cat.startswith("S")):
            return False
    return True


def _is_unk_marker(decoded: str) -> bool:
    """Return True if *decoded* contains a [UNK:…] pattern."""
    return bool(_UNK_MARKER_RE.search(decoded))


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _is_punct_or_symbol(ch: str) -> bool:
    """Return True iff *ch* is a Unicode punctuation or symbol character.

    Combining marks (category Mn, Mc, Me) return False — they must remain
    attached to their base glyph.
    """
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _split_punctuation(token: str) -> list[str]:
    """Split leading/trailing punctuation from a whitespace token.

    Mirrors VocabExtractor._split_punctuation semantics so that surface words
    extracted from decode_tokens() output are comparable with attested_vocab
    produced by VocabExtractor.
    """
    if not token:
        return []

    lead_end = 0
    while lead_end < len(token) and _is_punct_or_symbol(token[lead_end]):
        lead_end += 1

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


def _surface_words_from_text(text: str) -> list[str]:
    """Split a detokenised text into surface words using VocabExtractor semantics.

    Algorithm (mirrors VocabExtractor.extract):
      1. NFKC-normalise the full text.
      2. Split on Unicode whitespace.
      3. Separate leading/trailing punctuation/symbols from each token.
      4. Return non-empty, non-layout sub-tokens.

    This function is used for the surface-composition check in TokenAuditor.audit()
    after full-sequence decode_tokens() to handle BPE and byte-level tokenisers
    correctly.
    """
    nfkc_text = _nfkc(text)
    words: list[str] = []
    for raw_tok in nfkc_text.split():
        if not raw_tok:
            continue
        for sub in _split_punctuation(raw_tok):
            if sub and not _is_layout(sub):
                words.append(sub)
    return words


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class TokenAuditor:
    """Post-decode token-ID and surface-composition provenance auditor.

    See module docstring for the full specification.
    """

    # ------------------------------------------------------------------
    # build_license
    # ------------------------------------------------------------------

    def build_license(
        self,
        examples: Iterable[AlignedExample],
        backend: BackendProtocol,
    ) -> TokenLicense:
        """Build a token-ID licence from a collection of selected examples.

        For each example, calls ``backend.tokenize(example.target)`` and maps
        every returned token ID to the example's ``verse_idx``.

        Parameters
        ----------
        examples:
            Iterable of ``AlignedExample`` objects (the selected few-shot set).
        backend:
            Backend used to tokenize each ``target`` string.

        Returns
        -------
        TokenLicense
            ``{token_id: [verse_idx, …]}`` mapping.  A token ID that appears
            in *N* different example targets will have *N* verse_idx entries
            (duplicates within the same example are collapsed).
        """
        licence: TokenLicense = {}
        for example in examples:
            ids = backend.tokenize(example.target)
            seen_in_this_example: set[int] = set()
            for tid in ids:
                if tid in seen_in_this_example:
                    continue  # already recorded this verse_idx for this ID
                seen_in_this_example.add(tid)
                if tid not in licence:
                    licence[tid] = []
                if example.verse_idx not in licence[tid]:
                    licence[tid].append(example.verse_idx)
        return licence

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------

    def audit(
        self,
        token_ids: list[int],
        attested_vocab: frozenset[str],
        backend: BackendProtocol,
        *,
        examples: Optional[list[AlignedExample]] = None,
        license: Optional[TokenLicense] = None,
    ) -> tuple[TokenAuditResult, TokenLicense]:
        """Audit generated token IDs against the example provenance licence.

        Parameters
        ----------
        token_ids:
            The ``token_ids`` from a ``GenerationResult``.
        attested_vocab:
            The surface-level attested vocabulary for the request (from
            ``VocabExtractor.extract``).
        backend:
            Backend whose ``decode_token`` is called for each ID.
        examples:
            The selected few-shot examples.  Required if *license* is not
            supplied; ``build_license`` is called internally.
        license:
            A precomputed ``TokenLicense`` (from a previous ``build_license``
            call).  If supplied, *examples* is ignored and ``build_license``
            is **not** called.

        Returns
        -------
        (TokenAuditResult, TokenLicense)
            • ``TokenAuditResult`` — ``passed`` flag and ``violations`` list.
            • ``TokenLicense`` — provenance sub-map for IDs seen during this
              audit (non-layout tokens only).  Useful for downstream logging.

        Raises
        ------
        ValueError
            If neither *examples* nor *license* is supplied.
        """
        if license is None and examples is None:
            raise ValueError(
                "Either 'examples' or 'license' must be supplied to TokenAuditor.audit()."
            )

        # Build (or re-use) the token-ID licence.
        if license is not None:
            full_licence: TokenLicense = license
        else:
            assert examples is not None  # narrowing for type checker
            full_licence = self.build_license(examples, backend)

        if not token_ids:
            return TokenAuditResult(passed=True, violations=[]), {}

        # ── Step 1: Decode every token ID ───────────────────────────────────
        decoded_map: dict[int, str] = {}  # token_id → decoded surface (once)
        for tid in token_ids:
            if tid not in decoded_map:
                decoded_map[tid] = backend.decode_token(tid)

        violations: list[str] = []
        seen_provenance: TokenLicense = {}  # provenance for *this* audit call

        # ── Step 2: Per-token checks ─────────────────────────────────────────
        for tid in token_ids:
            decoded = decoded_map[tid]

            # --- 2a. UNK-marker check (invariant I3) ---
            if _is_unk_marker(decoded):
                violations.append(
                    f"token_id={tid} decoded={decoded!r} is a [UNK:…] marker "
                    f"in model output (invariant I3 violation)"
                )
                continue

            # --- 2b. Layout narrow rule (whitespace / punctuation) ---
            if _is_layout(decoded):
                # Layout tokens pass unconditionally; not added to provenance.
                continue

            # --- 2c. ID-level licence check ---
            if tid not in full_licence:
                violations.append(
                    f"token_id={tid} decoded={decoded!r} not in token licence "
                    f"(no example target produced this ID)"
                )
                continue

            # ID is licenced — record provenance for this audit call.
            if tid not in seen_provenance:
                seen_provenance[tid] = list(full_licence[tid])
            else:
                # Merge (unlikely to differ but be safe).
                for v in full_licence[tid]:
                    if v not in seen_provenance[tid]:
                        seen_provenance[tid].append(v)

        # ── Step 3: Surface-composition check ───────────────────────────────
        # Use backend.decode_tokens() (full-sequence detokenisation) for the
        # licensed token IDs.  This is required for BPE tokenisers (Qwen3.5)
        # and byte-level tokenisers where individual decode_token() calls
        # produce partial/garbled output.
        # Detokenise the complete sequence. Filtering out unlicensed or layout
        # IDs before this step would corrupt whitespace and byte-level Unicode
        # reconstruction; those IDs are already handled by Step 2.
        # Reconstruct the complete generated sequence, including layout and any
        # unlicensed IDs. Removing IDs before detokenisation can erase spaces or
        # byte fragments and falsely merge adjacent attested words. ID-level
        # violations are still reported independently above.
        # Surface recombination is only meaningful when every lexical ID has a
        # provenance licence. If any lexical ID is unlicensed, Step 2 already
        # makes the audit fail loudly; skip Step 3 to avoid duplicate/misattributed
        # surface violations while still allowing unlicensed layout IDs.
        has_unlicensed_lexical_id = any(
            tid not in full_licence and not _is_layout(decoded_map[tid])
            for tid in token_ids
        )
        if token_ids and not has_unlicensed_lexical_id:
            decode_tokens = getattr(backend, "decode_tokens", None)
            if callable(decode_tokens):
                # Required path for real BPE/byte-level backends.
                full_surface_text = decode_tokens(token_ids)
            else:
                # Compatibility path for narrow test doubles and legacy custom
                # backends. SentencePiece markers preserve subword boundaries;
                # otherwise conservatively treat each decoded piece as a word.
                pieces = [decoded_map[tid] for tid in token_ids]
                if any(_WORD_BOUNDARY in piece for piece in pieces):
                    full_surface_text = "".join(pieces).replace(_WORD_BOUNDARY, " ")
                else:
                    full_surface_text = " ".join(pieces)
            surface_words = _surface_words_from_text(full_surface_text)
        else:
            surface_words = []

        for word in surface_words:
            # _surface_words_from_text already applies NFKC and filters layout;
            # apply NFKC again defensively for the attested_vocab lookup.
            nfkc_word = _nfkc(word)
            if not nfkc_word:
                continue
            if nfkc_word not in attested_vocab:
                violations.append(
                    f"surface word {nfkc_word!r} is not in attested_vocab "
                    f"(subword recombination produced an unattested word)"
                )

        passed = len(violations) == 0
        return TokenAuditResult(passed=passed, violations=violations), seen_provenance
