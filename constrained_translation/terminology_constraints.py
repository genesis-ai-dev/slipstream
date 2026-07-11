"""constrained_translation.terminology_constraints — Task 4b.

Grammar + triggered terminology constraint composition.

Design
------
1. ``GlossaryEntry`` — stable, frozen, hashable entry with:
   - ``entry_id``: str identifier
   - ``source_trigger``: source phrase (may be multi-word); validated non-empty
   - ``target_phrase``: human target phrase; validated non-empty
   - ``provenance``: always ``"human_glossary"`` (validated explicitly, not inferred)

2. ``TerminologyConstraintComposer.resolve_active(source_text, glossary)``
   Determines which glossary entries are triggered by *source_text*.  Matching
   is done on NFKC-normalised, lowercased, punctuation-stripped token lists
   (same algorithm as ``text_normalize.normalize_source_units``).  A trigger
   fires if and only if its normalised token sequence appears as a **contiguous
   subsequence** in the normalised source tokens — never as an arbitrary
   substring.  Unicode and RTL safe.  No target-side signal.

3. ``TerminologyConstraintComposer.compose(...)``
   For each active entry:
   - Surface license is derived by NFKC-normalising the human ``target_phrase``
     text directly (whitespace split + leading/trailing punctuation separation,
     same as VocabExtractor).  Per-token decode is **never** used for surface
     extraction.
   - All token IDs from ``encode_fn(target_phrase)`` are added to the combined
     ID license.  IDs may only be excluded via an explicitly injected
     ``is_layout_id_fn`` classifier predicate — never by per-ID decoded
     whitespace guesses.  Default: include all encoded IDs.
   - Round-trip verification: ``decode_sequence_fn(encode_fn(phrase))`` must
     match ``phrase`` modulo NFKC + whitespace normalisation.  Failure →
     ``UnreachablePhraseError``.
   - Maintains ``surface_provenance`` and ``id_provenance`` (read-only
     MappingProxyType) mapping each surface / ID to the frozenset of glossary
     entry IDs that contributed it.
   - Returns a frozen ``ConstraintPlan`` with combined licenses, forced phrases,
     active entry IDs, and ``gbs_required`` flag.

4. Non-mutation — base ``base_surfaces`` and ``base_ids`` frozensets are never
   modified; new frozensets are produced.

5. Multiple overlapping active entries are handled deterministically: surfaces
   and IDs are unioned (deduplicated), forced phrases are deduplicated, but
   provenance is preserved for ALL contributors.

6. Grammar-only identity — if no entry is triggered, the returned plan has
   ``licensed_surfaces == base_surfaces``, ``licensed_token_ids == base_ids``,
   ``forced_phrases == ()``, ``active_entry_ids == ()``, and
   ``gbs_required == False``.  The encoder is NOT called in this case.

7. Capability pre-flight — if any entry is active and
   ``backend_supports_grammar_gbs`` is False (the default, as vLLM does not
   currently support grammar+GBS composition), a ``MissingCapabilityError`` is
   raised immediately.  Never silently drops grammar, terminology, or falls back.

   Task-10 gate: live vLLM grammar+GBS support is out of scope here.

8. Reachability pre-flight — for each forced phrase:
   - Its encoding must be non-empty (else ``UnreachablePhraseError``).
   - ``decode_sequence_fn(encoded_ids)`` must match the phrase modulo NFKC +
     whitespace normalisation (else ``UnreachablePhraseError``).
   - If a ``grammar_reachability_fn`` callback is supplied it is called with a
     frozen ``PhraseReachabilityRequest`` (phrase, encoded_ids, phrase_surfaces,
     combined licensed surfaces/IDs, allowed separators/punctuation).  If it
     returns False, ``UnreachablePhraseError`` is raised.

9. GBS guarantees **inclusion** only, not alignment.  Do not claim vLLM
   currently supports grammar + GBS composition.

10. ``ConstraintPlan`` is immutable: frozen dataclass; ``forced_phrases`` and
    ``active_entry_ids`` are tuples; ``surface_provenance`` and
    ``id_provenance`` are read-only ``MappingProxyType`` views.

Typed errors
------------
* ``MissingCapabilityError`` — backend lacks grammar+GBS composition.
* ``UnreachablePhraseError`` — a forced phrase cannot be reached (empty
  encoding, round-trip mismatch, or reachability callback returned False).

No runner, no policies, no CLI, no plots.  Pure and minimal.
"""
from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Optional

from constrained_translation.text_normalize import normalize_source_units
from constrained_translation.vocab_extractor import _split_punctuation


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------

class MissingCapabilityError(Exception):
    """Raised when the backend lacks grammar + GBS composition capability.

    Active terminology entries require GBS (guided beam search / beam-search
    with forced phrases) composed with the token-license grammar.  vLLM does
    NOT currently support this composition; that is a Task-10 gate.

    Never silently fall back — always raise this error loudly.
    """


class UnreachablePhraseError(Exception):
    """Raised when a forced target phrase cannot be reached.

    Reasons include:
    - The phrase encodes to an empty token list (empty encoding).
    - The full-sequence decode of the encoded IDs does not round-trip back to
      the human phrase (modulo NFKC + whitespace normalisation).
    - A supplied ``grammar_reachability_fn`` returns False for the phrase.

    Attributes
    ----------
    phrase : str
        The forced phrase that failed the reachability check.
    reason : str
        Human-readable explanation of why the phrase is unreachable.
    """

    def __init__(self, phrase: str, reason: str) -> None:
        super().__init__(f"Unreachable forced phrase {phrase!r}: {reason}")
        self.phrase = phrase
        self.reason = reason


# ---------------------------------------------------------------------------
# GlossaryEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlossaryEntry:
    """A single human-glossary entry mapping a source trigger to a target phrase.

    Attributes
    ----------
    entry_id : str
        Stable, unique identifier for this entry.
    source_trigger : str
        Source-side phrase (possibly multi-word) that activates this entry.
        Matched after NFKC normalisation, lowercasing, and punctuation removal.
        Must be non-empty after stripping whitespace.
    target_phrase : str
        Human-authored target phrase to force in the translation output.
        Must be non-empty after stripping whitespace.
    provenance : str
        Always ``"human_glossary"``.  Explicit, not inferred.  Any other value
        raises ``ValueError`` on construction.

    Notes
    -----
    Frozen (hashable) — safe to use in sets and as dict keys.
    """

    entry_id: str
    source_trigger: str
    target_phrase: str
    provenance: str = "human_glossary"

    def __post_init__(self) -> None:
        if not self.source_trigger or not self.source_trigger.strip():
            raise ValueError(
                f"GlossaryEntry.source_trigger must be non-empty (entry_id={self.entry_id!r})"
            )
        if not self.target_phrase or not self.target_phrase.strip():
            raise ValueError(
                f"GlossaryEntry.target_phrase must be non-empty (entry_id={self.entry_id!r})"
            )
        if self.provenance != "human_glossary":
            raise ValueError(
                f"GlossaryEntry.provenance must be 'human_glossary'; "
                f"got {self.provenance!r} (entry_id={self.entry_id!r})"
            )


# ---------------------------------------------------------------------------
# PhraseReachabilityRequest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhraseReachabilityRequest:
    """Immutable context bundle passed to the grammar reachability callback.

    The callback ``grammar_reachability_fn`` receives one of these and returns
    True iff the compiled grammar can reach the phrase given the context.

    Attributes
    ----------
    phrase : str
        The forced target phrase being checked.
    encoded_ids : tuple[int, ...]
        Token IDs produced by ``encode_fn(phrase)``.
    phrase_surfaces : frozenset[str]
        Lexical surface units extracted from *phrase* (human-text level, same
        algorithm as VocabExtractor — NFKC + whitespace split + punct split).
    licensed_surfaces : frozenset[str]
        Combined licensed surfaces (base ∪ glossary-derived) at the time of
        the reachability check.
    licensed_token_ids : frozenset[int]
        Combined licensed token IDs (base ∪ glossary-derived) at the time of
        the reachability check.
    allowed_separators : frozenset[str]
        Explicit set of allowed word-separator surface strings (e.g. spaces,
        hyphens) declared by the caller.  Default empty.
    allowed_punctuation : frozenset[str]
        Explicit set of allowed punctuation surface strings declared by the
        caller.  Default empty.
    """

    phrase: str
    encoded_ids: tuple
    phrase_surfaces: frozenset
    licensed_surfaces: frozenset
    licensed_token_ids: frozenset
    allowed_separators: frozenset
    allowed_punctuation: frozenset


# ---------------------------------------------------------------------------
# ConstraintPlan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintPlan:
    """The composed constraint plan for one translation unit.

    GBS guarantees **inclusion** only, not alignment.  The forced phrases will
    appear somewhere in the generated output, but their precise position and
    word order are not guaranteed.  vLLM does NOT currently support grammar +
    GBS composition; live support is a Task-10 gate.

    This dataclass is **immutable** (frozen).  All collection fields are
    immutable: ``licensed_surfaces`` and ``licensed_token_ids`` are frozensets;
    ``forced_phrases`` and ``active_entry_ids`` are tuples; ``surface_provenance``
    and ``id_provenance`` are read-only ``MappingProxyType`` views.

    Attributes
    ----------
    licensed_surfaces : frozenset[str]
        Combined set of licensed surface strings (base ∪ glossary-derived).
        Surfaces are human lexical units extracted from the target phrase text
        (NFKC + whitespace + punct split), not per-token decoded strings.
    licensed_token_ids : frozenset[int]
        Combined set of licensed token IDs (base ∪ glossary-derived).
        Includes all IDs from ``encode_fn(target_phrase)`` unless an explicit
        ``is_layout_id_fn`` predicate excludes them.
    forced_phrases : tuple[str, ...]
        Deduplicated tuple of target phrases to force via GBS.  Empty when no
        entries are active.
    active_entry_ids : tuple[str, ...]
        Ordered tuple of entry IDs that were triggered for this source segment.
        Empty when no entries are active.
    gbs_required : bool
        True iff at least one glossary entry is active.  When False, the backend
        GBS capability is irrelevant and no GBS request is made.
    surface_provenance : MappingProxyType[str, frozenset[str]]
        Read-only map from each glossary-derived surface string to the frozenset
        of ``entry_id`` strings that contributed it.
    id_provenance : MappingProxyType[int, frozenset[str]]
        Read-only map from each glossary-derived token ID to the frozenset of
        ``entry_id`` strings that contributed it.
    """

    licensed_surfaces: frozenset
    licensed_token_ids: frozenset
    forced_phrases: tuple
    active_entry_ids: tuple
    gbs_required: bool
    surface_provenance: MappingProxyType
    id_provenance: MappingProxyType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_trigger(trigger: str) -> list[str]:
    """Normalise a source trigger to a list of comparable tokens.

    Uses the same pipeline as ``normalize_source_units`` so trigger matching
    is consistent with source-segment normalisation.
    """
    return normalize_source_units(trigger)


def _is_sublist(needle: list[str], haystack: list[str]) -> bool:
    """Return True iff *needle* appears as a contiguous subsequence in *haystack*.

    Both lists should already be normalised.  Empty needle always matches.
    """
    if not needle:
        return True
    n, h = len(needle), len(haystack)
    for i in range(h - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _extract_phrase_surfaces(phrase: str) -> frozenset:
    """Extract human lexical surface units from a target phrase.

    Uses the same algorithm as VocabExtractor:
    1. Split on Unicode whitespace.
    2. NFKC-normalise each whitespace-delimited token.
    3. Separate leading/trailing punctuation into individual entries.
    4. Return frozenset of non-empty surface strings.

    This operates purely on the human-readable phrase text and does NOT
    call any tokenizer decode function.

    Parameters
    ----------
    phrase : str
        Human-authored target phrase (may be multi-word).

    Returns
    -------
    frozenset[str]
        NFKC-normalised lexical surface units derived from *phrase*.
    """
    tokens: set = set()
    for raw_tok in phrase.split():
        if not raw_tok:
            continue
        nfkc_tok = unicodedata.normalize("NFKC", raw_tok)
        for sub in _split_punctuation(nfkc_tok):
            if sub:
                tokens.add(sub)
    return frozenset(tokens)


def _normalize_for_roundtrip(text: str) -> str:
    """Normalise text for round-trip comparison: NFKC + strip whitespace."""
    return unicodedata.normalize("NFKC", text).strip()


# ---------------------------------------------------------------------------
# TerminologyConstraintComposer
# ---------------------------------------------------------------------------

class TerminologyConstraintComposer:
    """Compose terminology constraints with grammar-licensed translation requests.

    Parameters
    ----------
    encode_fn : Callable[[str], list[int]]
        Injected tokenizer: encodes a target phrase string to a list of integer
        token IDs.  Called only for active entries.
    decode_sequence_fn : Callable[[list[int]], str]
        Injected full-sequence detokenizer: decodes the complete list of token
        IDs produced by ``encode_fn`` back to a string.  Used for round-trip
        verification only.  Byte/BPE fragments are handled correctly because the
        full sequence is decoded in one call — not one per token.  Empty list
        must return ``""``.
    is_layout_id_fn : Callable[[int], bool] | None
        Optional predicate: returns True iff a token ID is a structural/layout/
        control token that should be excluded from the lexical ID license.  When
        None (the default), all token IDs from ``encode_fn`` are licensed.
        Whitespace guessing via per-token decode is explicitly forbidden — only
        this explicit predicate may exclude IDs.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], list],
        decode_sequence_fn: Callable[[list], str],
        is_layout_id_fn: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self._encode_fn = encode_fn
        self._decode_sequence_fn = decode_sequence_fn
        self._is_layout_id_fn = is_layout_id_fn

    def resolve_active(
        self,
        source_text: str,
        glossary: list,
    ) -> list:
        """Return the glossary entries whose triggers are active in *source_text*.

        Matching is deterministic and contiguous:

        1. Normalise the source text to a token list via
           ``normalize_source_units``.
        2. For each entry, normalise its ``source_trigger`` the same way.
        3. The entry is active iff the normalised trigger tokens appear as a
           **contiguous subsequence** in the normalised source tokens.

        Ordering of the returned list follows the input *glossary* order
        (deterministic).

        Parameters
        ----------
        source_text : str
            Raw source segment text.
        glossary : list[GlossaryEntry]
            All candidate glossary entries.

        Returns
        -------
        list[GlossaryEntry]
            The active entries, in glossary order.
        """
        if not source_text.strip() or not glossary:
            return []

        src_tokens = normalize_source_units(source_text)
        active = []
        for entry in glossary:
            trigger_tokens = _normalize_trigger(entry.source_trigger)
            if trigger_tokens and _is_sublist(trigger_tokens, src_tokens):
                active.append(entry)
        return active

    def compose(
        self,
        source_text: str,
        glossary: list,
        base_surfaces: frozenset,
        base_ids: frozenset,
        backend_supports_grammar_gbs: bool = False,
        grammar_reachability_fn: Optional[Callable] = None,
        allowed_separators: frozenset = frozenset(),
        allowed_punctuation: frozenset = frozenset(),
    ) -> ConstraintPlan:
        """Compose terminology constraints into a ``ConstraintPlan``.

        Steps
        -----
        1. Resolve active entries for *source_text*.
        2. If no entries are active → return grammar-only identity plan (base
           licenses unchanged, no GBS, no forced phrases).
        3. Capability pre-flight: active + ``backend_supports_grammar_gbs=False``
           → raise ``MissingCapabilityError``.
        4. For each active entry:
           a. Encode the target phrase via ``encode_fn``.
           b. Check for empty encoding → raise ``UnreachablePhraseError``.
           c. Verify round-trip: ``decode_sequence_fn(ids)`` must match phrase
              modulo NFKC + whitespace normalisation → else ``UnreachablePhraseError``.
           d. Extract surfaces from the human phrase text (NFKC + whitespace +
              punct split, same as VocabExtractor).
           e. License all encoded IDs unless ``is_layout_id_fn`` excludes them.
           f. Track provenance for surfaces and IDs.
        5. Reachability callback pre-flight: if ``grammar_reachability_fn`` is
           supplied and returns False for a ``PhraseReachabilityRequest`` →
           raise ``UnreachablePhraseError``.
        6. Return frozen ``ConstraintPlan`` with combined licenses, forced
           phrases, active entry IDs, and ``gbs_required=True``.

        Parameters
        ----------
        source_text : str
            Raw source segment text.
        glossary : list[GlossaryEntry]
            All candidate glossary entries.
        base_surfaces : frozenset[str]
            Grammar-licensed surface strings from the base pipeline (not mutated).
        base_ids : frozenset[int]
            Grammar-licensed token IDs from the base pipeline (not mutated).
        backend_supports_grammar_gbs : bool
            True iff the backend supports grammar + GBS composition.  Default
            False — vLLM does not currently support this; live support is Task 10.
        grammar_reachability_fn : Callable[[PhraseReachabilityRequest], bool] | None
            Optional callback: given a ``PhraseReachabilityRequest``, returns
            True iff the compiled grammar can reach the phrase.  If it returns
            False, an ``UnreachablePhraseError`` is raised.
        allowed_separators : frozenset[str]
            Explicit set of allowed separator surfaces forwarded to the
            reachability callback.  Default empty frozenset.
        allowed_punctuation : frozenset[str]
            Explicit set of allowed punctuation surfaces forwarded to the
            reachability callback.  Default empty frozenset.

        Returns
        -------
        ConstraintPlan

        Raises
        ------
        MissingCapabilityError
            Active entries + ``backend_supports_grammar_gbs=False``.
        UnreachablePhraseError
            A forced phrase encodes to empty, fails round-trip, or the
            reachability callback returns False for it.
        """
        active = self.resolve_active(source_text, glossary)

        # §6 — Grammar-only identity
        if not active:
            return ConstraintPlan(
                licensed_surfaces=base_surfaces,
                licensed_token_ids=base_ids,
                forced_phrases=(),
                active_entry_ids=(),
                gbs_required=False,
                surface_provenance=MappingProxyType({}),
                id_provenance=MappingProxyType({}),
            )

        # §7 — Capability pre-flight
        if not backend_supports_grammar_gbs:
            raise MissingCapabilityError(
                "Active terminology entries require grammar + GBS composition, "
                "but the backend does not support it.  "
                "vLLM grammar + GBS composition is not yet available; "
                "this is a Task-10 gate.  "
                "Set backend_supports_grammar_gbs=True only when the backend "
                "explicitly declares this capability."
            )

        # §3/§4 — Encode each active entry's target phrase, build license union
        # with provenance.  Process in deterministic (glossary) order.

        # Accumulate mutable dicts then freeze.
        extra_surfaces: dict = {}   # surface -> {entry_id, ...}
        extra_ids: dict = {}        # token_id -> {entry_id, ...}

        seen_phrases: dict = {}     # phrase -> (token_ids, phrase_surfaces)
        forced_phrases_set: dict = {}  # ordered dedup via insertion-order dict
        active_entry_ids: list = []

        for entry in active:
            active_entry_ids.append(entry.entry_id)

            phrase = entry.target_phrase
            forced_phrases_set[phrase] = None  # dedup, preserve order

            # Encode (cache to avoid repeated calls for same phrase)
            if phrase not in seen_phrases:
                token_ids = self._encode_fn(phrase)

                # Reachability: empty encoding
                if not token_ids:
                    raise UnreachablePhraseError(
                        phrase=phrase,
                        reason=(
                            f"encode_fn returned empty token list for phrase {phrase!r} "
                            f"(entry_id={entry.entry_id!r})"
                        ),
                    )

                # Round-trip verification via full-sequence decode
                decoded = self._decode_sequence_fn(token_ids)
                if _normalize_for_roundtrip(decoded) != _normalize_for_roundtrip(phrase):
                    raise UnreachablePhraseError(
                        phrase=phrase,
                        reason=(
                            f"round-trip decode mismatch for phrase {phrase!r}: "
                            f"encode_fn then decode_sequence_fn gives {decoded!r}, "
                            f"expected ~{phrase!r} (entry_id={entry.entry_id!r})"
                        ),
                    )

                # Extract human-text surfaces from the phrase (NOT per-token decode)
                phrase_surfs = _extract_phrase_surfaces(phrase)
                seen_phrases[phrase] = (token_ids, phrase_surfs)
            else:
                token_ids, phrase_surfs = seen_phrases[phrase]

            # Surface provenance — keyed by human word units from phrase text
            for surf in phrase_surfs:
                if surf not in extra_surfaces:
                    extra_surfaces[surf] = set()
                extra_surfaces[surf].add(entry.entry_id)

            # ID provenance — all encoded IDs unless is_layout_id_fn excludes them
            for tid in token_ids:
                if self._is_layout_id_fn is not None and self._is_layout_id_fn(tid):
                    continue  # explicit layout/structural ID exclusion
                if tid not in extra_ids:
                    extra_ids[tid] = set()
                extra_ids[tid].add(entry.entry_id)

        # Build combined (immutable) licenses — never mutate base sets
        combined_surfaces: frozenset = base_surfaces | frozenset(extra_surfaces.keys())
        combined_ids: frozenset = base_ids | frozenset(extra_ids.keys())

        # Freeze provenance dicts
        surface_provenance_raw: dict = {
            s: frozenset(eids) for s, eids in extra_surfaces.items()
        }
        id_provenance_raw: dict = {
            tid: frozenset(eids) for tid, eids in extra_ids.items()
        }

        # §8 — Reachability callback pre-flight (after encoding, per phrase)
        if grammar_reachability_fn is not None:
            for phrase in forced_phrases_set:
                token_ids, phrase_surfs = seen_phrases[phrase]
                req = PhraseReachabilityRequest(
                    phrase=phrase,
                    encoded_ids=tuple(token_ids),
                    phrase_surfaces=phrase_surfs,
                    licensed_surfaces=combined_surfaces,
                    licensed_token_ids=combined_ids,
                    allowed_separators=allowed_separators,
                    allowed_punctuation=allowed_punctuation,
                )
                if not grammar_reachability_fn(req):
                    raise UnreachablePhraseError(
                        phrase=phrase,
                        reason=(
                            f"grammar_reachability_fn returned False for phrase {phrase!r}"
                        ),
                    )

        return ConstraintPlan(
            licensed_surfaces=combined_surfaces,
            licensed_token_ids=combined_ids,
            forced_phrases=tuple(forced_phrases_set.keys()),
            active_entry_ids=tuple(active_entry_ids),
            gbs_required=True,
            surface_provenance=MappingProxyType(surface_provenance_raw),
            id_provenance=MappingProxyType(id_provenance_raw),
        )
