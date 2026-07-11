"""constrained_translation.terminology_constraints — Task 4b.

Grammar + triggered terminology constraint composition.

Design
------
1. ``GlossaryEntry`` — stable, frozen, hashable entry with:
   - ``entry_id``: str identifier
   - ``source_trigger``: source phrase (may be multi-word); validated non-empty
   - ``target_phrase``: human target phrase; validated non-empty
   - ``provenance``: always ``"human_glossary"`` (explicit, not inferred)

2. ``TerminologyConstraintComposer.resolve_active(source_text, glossary)``
   Determines which glossary entries are triggered by *source_text*.  Matching
   is done on NFKC-normalised, lowercased, punctuation-stripped token lists
   (same algorithm as ``text_normalize.normalize_source_units``).  A trigger
   fires if and only if its normalised token sequence appears as a **contiguous
   subsequence** in the normalised source tokens — never as an arbitrary
   substring.  Unicode and RTL safe.  No target-side signal.

3. ``TerminologyConstraintComposer.compose(...)``
   For each active entry:
   - Tokenizes the human target phrase via the injected ``encode_fn``.
   - Adds decoded surface strings (via ``decode_fn``) and all lexical token IDs
     (non-whitespace) to the combined license BEFORE grammar compilation.
   - Maintains ``surface_provenance`` and ``id_provenance`` dicts mapping each
     added surface / ID to the set of glossary entry IDs that contributed it.
   - Returns a ``ConstraintPlan`` with the combined licenses, forced phrases,
     active entry IDs, and ``gbs_required`` flag.

4. Non-mutation — base ``base_surfaces`` and ``base_ids`` frozensets are never
   modified; new frozensets are produced.

5. Multiple overlapping active entries are handled deterministically: surfaces
   and IDs are unioned (deduplicated), forced phrases are deduplicated, but
   provenance is preserved for ALL contributors.

6. Grammar-only identity — if no entry is triggered, the returned plan has
   ``licensed_surfaces == base_surfaces``, ``licensed_token_ids == base_ids``,
   ``forced_phrases == []``, ``active_entry_ids == []``, and
   ``gbs_required == False``.  The encoder is NOT called in this case.

7. Capability pre-flight — if any entry is active and
   ``backend_supports_grammar_gbs`` is False (the default, as vLLM does not
   currently support grammar+GBS composition), a ``MissingCapabilityError`` is
   raised immediately.  Never silently drops grammar, terminology, or falls back.

   Task-10 gate: live vLLM grammar+GBS support is out of scope here.

8. Reachability pre-flight — for each forced phrase:
   - Its encoding must be non-empty (else ``UnreachablePhraseError``).
   - If a ``grammar_reachability_fn`` callback is supplied and returns False for
     the phrase, ``UnreachablePhraseError`` is raised.

9. GBS guarantees **inclusion** only, not alignment.  Do not claim vLLM
   currently supports grammar + GBS composition.

Typed errors
------------
* ``MissingCapabilityError`` — backend lacks grammar+GBS composition.
* ``UnreachablePhraseError`` — a forced phrase cannot be reached (empty encoding
  or reachability callback returned False).

No runner, no policies, no CLI, no plots.  Pure and minimal.
"""
from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from constrained_translation.text_normalize import normalize_source_units


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
        Always ``"human_glossary"``.  Explicit, not inferred.

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


# ---------------------------------------------------------------------------
# ConstraintPlan
# ---------------------------------------------------------------------------

@dataclass
class ConstraintPlan:
    """The composed constraint plan for one translation unit.

    GBS guarantees **inclusion** only, not alignment.  The forced phrases will
    appear somewhere in the generated output, but their precise position and
    word order are not guaranteed.  vLLM does NOT currently support grammar +
    GBS composition; live support is a Task-10 gate.

    Attributes
    ----------
    licensed_surfaces : frozenset[str]
        Combined set of licensed surface strings (base ∪ glossary-derived).
        Lexical only — whitespace surfaces are excluded.
    licensed_token_ids : frozenset[int]
        Combined set of licensed token IDs (base ∪ glossary-derived).
        Lexical only — whitespace-only token IDs are excluded.
    forced_phrases : list[str]
        Deduplicated list of target phrases to force via GBS.  Empty when no
        entries are active.
    active_entry_ids : list[str]
        Ordered list of entry IDs that were triggered for this source segment.
        Empty when no entries are active.
    gbs_required : bool
        True iff at least one glossary entry is active.  When False, the backend
        GBS capability is irrelevant and no GBS request is made.
    surface_provenance : dict[str, frozenset[str]]
        Maps each glossary-derived surface string to the frozenset of
        ``entry_id`` strings that contributed it.
    id_provenance : dict[int, frozenset[str]]
        Maps each glossary-derived token ID to the frozenset of ``entry_id``
        strings that contributed it.
    """

    licensed_surfaces: frozenset[str]
    licensed_token_ids: frozenset[int]
    forced_phrases: list[str]
    active_entry_ids: list[str]
    gbs_required: bool
    surface_provenance: dict[str, frozenset[str]]
    id_provenance: dict[int, frozenset[str]]


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


_WS_RE = re.compile(r"^\s*$")


def _is_whitespace_only(text: str) -> bool:
    """Return True if *text* consists only of whitespace characters."""
    return bool(_WS_RE.match(text))


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
    decode_fn : Callable[[int], str]
        Injected detokenizer: decodes a single token ID to its surface string.
        Used to derive surface strings from encoded token IDs.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[int], str],
    ) -> None:
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn

    def resolve_active(
        self,
        source_text: str,
        glossary: list[GlossaryEntry],
    ) -> list[GlossaryEntry]:
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
        active: list[GlossaryEntry] = []
        for entry in glossary:
            trigger_tokens = _normalize_trigger(entry.source_trigger)
            if trigger_tokens and _is_sublist(trigger_tokens, src_tokens):
                active.append(entry)
        return active

    def compose(
        self,
        source_text: str,
        glossary: list[GlossaryEntry],
        base_surfaces: frozenset[str],
        base_ids: frozenset[int],
        backend_supports_grammar_gbs: bool = False,
        grammar_reachability_fn: Optional[Callable[[str], bool]] = None,
    ) -> ConstraintPlan:
        """Compose terminology constraints into a ``ConstraintPlan``.

        Steps
        -----
        1. Resolve active entries for *source_text*.
        2. If no entries are active → return grammar-only identity plan (base
           licenses unchanged, no GBS, no forced phrases).
        3. Capability pre-flight: active + ``backend_supports_grammar_gbs=False``
           → raise ``MissingCapabilityError``.
        4. For each active entry, encode the target phrase.  Build the union of
           lexical surfaces and IDs (whitespace excluded) with provenance tracking.
        5. Reachability pre-flight: empty encoding or callback returning False →
           raise ``UnreachablePhraseError``.
        6. Return ``ConstraintPlan`` with combined licenses, forced phrases,
           active entry IDs, and ``gbs_required=True``.

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
        grammar_reachability_fn : Callable[[str], bool] | None
            Optional callback: given a forced phrase string, returns True iff the
            compiled grammar can reach it.  If it returns False, an
            ``UnreachablePhraseError`` is raised.

        Returns
        -------
        ConstraintPlan

        Raises
        ------
        MissingCapabilityError
            Active entries + ``backend_supports_grammar_gbs=False``.
        UnreachablePhraseError
            A forced phrase encodes to empty, or the reachability callback
            returns False for it.
        """
        active = self.resolve_active(source_text, glossary)

        # §5 — Grammar-only identity
        if not active:
            return ConstraintPlan(
                licensed_surfaces=base_surfaces,
                licensed_token_ids=base_ids,
                forced_phrases=[],
                active_entry_ids=[],
                gbs_required=False,
                surface_provenance={},
                id_provenance={},
            )

        # §6 — Capability pre-flight
        if not backend_supports_grammar_gbs:
            raise MissingCapabilityError(
                "Active terminology entries require grammar + GBS composition, "
                "but the backend does not support it.  "
                "vLLM grammar + GBS composition is not yet available; "
                "this is a Task-10 gate.  "
                "Set backend_supports_grammar_gbs=True only when the backend "
                "explicitly declares this capability."
            )

        # §3 / §4 — Encode each active entry's target phrase, build license union
        # with provenance.  Process in deterministic (glossary) order.

        # Accumulate mutable dicts then freeze.
        extra_surfaces: dict[str, set[str]] = {}  # surface -> {entry_id, ...}
        extra_ids: dict[int, set[str]] = {}       # token_id -> {entry_id, ...}

        seen_phrases: dict[str, list[int]] = {}   # phrase -> token_ids (cache)
        forced_phrases_set: dict[str, None] = {}  # ordered dedup via insertion-order dict
        active_entry_ids: list[str] = []

        for entry in active:
            active_entry_ids.append(entry.entry_id)

            phrase = entry.target_phrase
            forced_phrases_set[phrase] = None  # dedup, preserve order

            # Encode (cache to avoid repeated calls for same phrase)
            if phrase not in seen_phrases:
                token_ids = self._encode_fn(phrase)
                seen_phrases[phrase] = token_ids
            else:
                token_ids = seen_phrases[phrase]

            # Reachability: empty encoding
            if not token_ids:
                raise UnreachablePhraseError(
                    phrase=phrase,
                    reason=(
                        f"encode_fn returned empty token list for phrase {phrase!r} "
                        f"(entry_id={entry.entry_id!r})"
                    ),
                )

            # Collect lexical surfaces and IDs (skip whitespace-only tokens)
            for tid in token_ids:
                surface = self._decode_fn(tid)
                if _is_whitespace_only(surface):
                    continue  # §3 — no added model / whitespace tokens
                # Surface provenance
                if surface not in extra_surfaces:
                    extra_surfaces[surface] = set()
                extra_surfaces[surface].add(entry.entry_id)
                # ID provenance
                if tid not in extra_ids:
                    extra_ids[tid] = set()
                extra_ids[tid].add(entry.entry_id)

        # §7 — Reachability callback pre-flight (after encoding, per phrase)
        if grammar_reachability_fn is not None:
            for phrase in forced_phrases_set:
                if not grammar_reachability_fn(phrase):
                    raise UnreachablePhraseError(
                        phrase=phrase,
                        reason=(
                            f"grammar_reachability_fn returned False for phrase {phrase!r}"
                        ),
                    )

        # Build combined (immutable) licenses — never mutate base sets
        combined_surfaces: frozenset[str] = base_surfaces | frozenset(extra_surfaces.keys())
        combined_ids: frozenset[int] = base_ids | frozenset(extra_ids.keys())

        # Freeze provenance dicts
        surface_provenance: dict[str, frozenset[str]] = {
            s: frozenset(eids) for s, eids in extra_surfaces.items()
        }
        id_provenance: dict[int, frozenset[str]] = {
            tid: frozenset(eids) for tid, eids in extra_ids.items()
        }

        return ConstraintPlan(
            licensed_surfaces=combined_surfaces,
            licensed_token_ids=combined_ids,
            forced_phrases=list(forced_phrases_set.keys()),
            active_entry_ids=active_entry_ids,
            gbs_required=True,
            surface_provenance=surface_provenance,
            id_provenance=id_provenance,
        )
