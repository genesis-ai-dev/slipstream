"""Tests for constrained_translation.terminology_constraints — Task 4b.

TDD: tests written BEFORE the implementation (RED/GREEN cycle).

Design assumptions
------------------
* Source-trigger matching is done on NFKC-normalised, lowercased, punct-stripped
  tokens (same algorithm as text_normalize.normalize_source_units).
* A trigger must match as a *contiguous subsequence* of the normalised source
  token list — never as an arbitrary substring of a concatenated string.
* The injected encode function is called ONLY for entries that are active.
* Base license frozensets are never mutated.
* Multiple overlapping active entries are deduplicated at the token-ID / surface
  level, but provenance (entry_id → surface/ID) is preserved for all contributors.
* GBS guarantees inclusion, not alignment.
* vLLM does NOT currently support grammar + GBS composition; that is Task 10.
  This module defines the plan/capability boundary only.

API notes (Task 4b tokenizer-safety migration)
----------------------------------------------
* ``TerminologyConstraintComposer`` now requires ``decode_sequence_fn``
  (full-sequence decode) instead of the old ``decode_fn`` (per-ID decode).
  Per-ID decode is explicitly forbidden for surface/ID extraction.
* Surfaces come from human-text phrase splitting (NFKC + whitespace + punct),
  never from per-token decoding.
* All encoded IDs are licensed unless an explicit ``is_layout_id_fn`` predicate
  excludes them (never whitespace guessing via per-ID decode).
* ``ConstraintPlan`` is frozen; ``forced_phrases`` and ``active_entry_ids`` are
  tuples; provenance maps are read-only ``MappingProxyType``.
* ``grammar_reachability_fn`` receives a ``PhraseReachabilityRequest`` object,
  not a bare string.
* Provenance field on ``GlossaryEntry`` must equal ``"human_glossary"`` exactly.
"""
from __future__ import annotations

import types
import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_mod():
    import constrained_translation.terminology_constraints as m
    return m


# ---------------------------------------------------------------------------
# Fake character-level tokenizer (pure, deterministic, offline)
# ---------------------------------------------------------------------------

def char_encode(text: str) -> list[int]:
    """Encode *text* as a list of Unicode code-point integers (character-level)."""
    return [ord(c) for c in text]


def char_decode_sequence(ids: list[int]) -> str:
    """Full-sequence decode: join all code-point integers back to a string."""
    return "".join(chr(i) for i in ids)


def make_spy_encode():
    """Return (spy_encode, calls) where calls is a list that records each call."""
    calls: list[str] = []

    def spy_encode(text: str) -> list[int]:
        calls.append(text)
        return char_encode(text)

    return spy_encode, calls


# ---------------------------------------------------------------------------
# §0 — API signature: new decode_sequence_fn parameter
# ---------------------------------------------------------------------------

class TestAPISignature:
    """Verify the new tokenizer-safe API surface."""

    def test_composer_accepts_decode_sequence_fn(self):
        """TerminologyConstraintComposer accepts decode_sequence_fn (not decode_fn)."""
        m = _import_mod()
        # Must not raise
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        assert c is not None

    def test_composer_rejects_decode_fn_keyword(self):
        """Old decode_fn keyword must raise TypeError (removed from API)."""
        m = _import_mod()
        with pytest.raises(TypeError):
            m.TerminologyConstraintComposer(
                encode_fn=char_encode,
                decode_fn=lambda tid: chr(tid),  # type: ignore[call-arg]
            )

    def test_composer_accepts_optional_is_layout_id_fn(self):
        """is_layout_id_fn is optional and defaults to None (include all IDs)."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
            is_layout_id_fn=None,
        )
        assert c is not None

    def test_composer_accepts_explicit_is_layout_id_fn(self):
        """is_layout_id_fn can be an explicit callable."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
            is_layout_id_fn=lambda tid: tid < 0,
        )
        assert c is not None


# ---------------------------------------------------------------------------
# §1 — GlossaryEntry construction and validation
# ---------------------------------------------------------------------------

class TestGlossaryEntry:
    def test_basic_construction(self):
        m = _import_mod()
        e = m.GlossaryEntry(
            entry_id="G001",
            source_trigger="God",
            target_phrase="Dieu",
        )
        assert e.entry_id == "G001"
        assert e.source_trigger == "God"
        assert e.target_phrase == "Dieu"
        assert e.provenance == "human_glossary"

    def test_provenance_default_is_human_glossary(self):
        m = _import_mod()
        e = m.GlossaryEntry(entry_id="G002", source_trigger="heaven", target_phrase="ciel")
        assert e.provenance == "human_glossary"

    def test_provenance_explicit_human_glossary(self):
        m = _import_mod()
        e = m.GlossaryEntry(
            entry_id="G003",
            source_trigger="earth",
            target_phrase="terre",
            provenance="human_glossary",
        )
        assert e.provenance == "human_glossary"

    def test_provenance_wrong_value_raises_value_error(self):
        """Provenance must be 'human_glossary' exactly — any other value raises."""
        m = _import_mod()
        with pytest.raises(ValueError, match="provenance"):
            m.GlossaryEntry(
                entry_id="G_bad",
                source_trigger="earth",
                target_phrase="terre",
                provenance="machine_translation",
            )

    def test_provenance_empty_string_raises_value_error(self):
        m = _import_mod()
        with pytest.raises(ValueError, match="provenance"):
            m.GlossaryEntry(
                entry_id="G_bad2",
                source_trigger="earth",
                target_phrase="terre",
                provenance="",
            )

    def test_empty_source_trigger_raises_value_error(self):
        m = _import_mod()
        with pytest.raises(ValueError, match="source_trigger"):
            m.GlossaryEntry(entry_id="G004", source_trigger="", target_phrase="terre")

    def test_whitespace_only_source_trigger_raises_value_error(self):
        m = _import_mod()
        with pytest.raises(ValueError, match="source_trigger"):
            m.GlossaryEntry(entry_id="G005", source_trigger="   ", target_phrase="terre")

    def test_empty_target_phrase_raises_value_error(self):
        m = _import_mod()
        with pytest.raises(ValueError, match="target_phrase"):
            m.GlossaryEntry(entry_id="G006", source_trigger="earth", target_phrase="")

    def test_whitespace_only_target_phrase_raises_value_error(self):
        m = _import_mod()
        with pytest.raises(ValueError, match="target_phrase"):
            m.GlossaryEntry(entry_id="G007", source_trigger="earth", target_phrase="  ")

    def test_entry_is_frozen_hashable(self):
        m = _import_mod()
        e = m.GlossaryEntry(entry_id="G008", source_trigger="light", target_phrase="lumière")
        # Frozen: attribute assignment must raise
        with pytest.raises((AttributeError, TypeError)):
            e.entry_id = "changed"  # type: ignore[misc]
        # Hashable
        assert isinstance(hash(e), int)

    def test_entries_in_set(self):
        m = _import_mod()
        e1 = m.GlossaryEntry(entry_id="G009", source_trigger="light", target_phrase="lumière")
        e2 = m.GlossaryEntry(entry_id="G010", source_trigger="dark", target_phrase="ténèbres")
        s = {e1, e2}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# §2 — Trigger resolution (active entry detection)
# ---------------------------------------------------------------------------

class TestTriggerResolution:
    """resolve_active(source_text, glossary) → list[GlossaryEntry] tests."""

    def _make_composer(self, m):
        return m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )

    def test_single_trigger_present(self):
        m = _import_mod()
        c = self._make_composer(m)
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heaven", "ciel"),
        ]
        active = c.resolve_active("In the beginning God created the heavens", entries)
        ids = {e.entry_id for e in active}
        assert "G1" in ids

    def test_trigger_not_present_excluded(self):
        m = _import_mod()
        c = self._make_composer(m)
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "serpent", "serpent"),
        ]
        active = c.resolve_active("In the beginning God created the heavens", entries)
        ids = {e.entry_id for e in active}
        assert "G2" not in ids

    def test_empty_source_returns_no_active(self):
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        active = c.resolve_active("", entries)
        assert active == []

    def test_empty_glossary_returns_empty(self):
        m = _import_mod()
        c = self._make_composer(m)
        active = c.resolve_active("In the beginning God created", [])
        assert active == []

    def test_multiword_trigger_contiguous_match(self):
        """Multi-word trigger must appear as a contiguous normalised phrase."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "God created", "Dieu créa")]
        # Present: "God created the heavens"
        active = c.resolve_active("God created the heavens", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_multiword_trigger_noncontiguous_no_match(self):
        """Non-contiguous occurrence of trigger words must NOT match."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "God created", "Dieu créa")]
        # "God" and "created" both present but not adjacent
        active = c.resolve_active("God the creator who created all things", entries)
        assert not any(e.entry_id == "G1" for e in active)

    def test_phrase_boundary_no_partial_word_match(self):
        """'god' trigger must NOT fire on 'goddess'."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "god", "Dieu")]
        active = c.resolve_active("The goddess Athena", entries)
        # 'goddess' normalises to 'goddess', not 'god'
        assert not any(e.entry_id == "G1" for e in active)

    def test_case_insensitive_trigger_match(self):
        """Source trigger is matched case-insensitively after normalisation."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "GOD", "Dieu")]
        active = c.resolve_active("In the beginning god created", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_punctuation_in_source_ignored_for_matching(self):
        """Punctuation in source is stripped before trigger matching."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        active = c.resolve_active("God, who created the heavens!", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_punctuation_in_trigger_ignored_for_matching(self):
        """Punctuation in trigger is stripped before normalisation."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "God,", "Dieu")]
        active = c.resolve_active("God created the heavens", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_unicode_rtl_trigger(self):
        """Unicode/RTL text works as both source and trigger."""
        m = _import_mod()
        c = self._make_composer(m)
        # Hebrew-like test: trigger word in source
        entries = [m.GlossaryEntry("G1", "אלהים", "Dieu")]
        active = c.resolve_active("בראשית ברא אלהים את השמים", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_unicode_rtl_trigger_absent(self):
        """Unicode/RTL trigger not present -> not active."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [m.GlossaryEntry("G1", "אלהים", "Dieu")]
        active = c.resolve_active("In the beginning God created", entries)
        assert not any(e.entry_id == "G1" for e in active)

    def test_nfkc_normalization_in_trigger_matching(self):
        """NFKC equivalents in source and trigger are matched correctly."""
        m = _import_mod()
        c = self._make_composer(m)
        # U+2126 OHM SIGN normalises to U+03A9 GREEK CAPITAL LETTER OMEGA under NFKC
        entries = [m.GlossaryEntry("G1", "\u03A9mega", "omega")]
        # Use the non-normalised form in source
        active = c.resolve_active("\u2126mega is a unit", entries)
        assert any(e.entry_id == "G1" for e in active)

    def test_multiple_active_entries(self):
        """Multiple entries can be active simultaneously."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
            m.GlossaryEntry("G3", "serpent", "serpent"),
        ]
        active = c.resolve_active("In the beginning God created the heavens", entries)
        ids = {e.entry_id for e in active}
        assert "G1" in ids
        assert "G2" in ids
        assert "G3" not in ids

    def test_resolve_active_deterministic_order(self):
        """resolve_active result ordering is deterministic across calls."""
        m = _import_mod()
        c = self._make_composer(m)
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
        ]
        source = "In the beginning God created the heavens"
        r1 = [e.entry_id for e in c.resolve_active(source, entries)]
        r2 = [e.entry_id for e in c.resolve_active(source, entries)]
        assert r1 == r2


# ---------------------------------------------------------------------------
# §3 — Encoder call behaviour
# ---------------------------------------------------------------------------

class TestEncoderCallBehaviour:
    def test_no_encoder_call_when_no_active_entries(self):
        """Encoder must NOT be called when no triggers are active."""
        m = _import_mod()
        spy, calls = make_spy_encode()
        c = m.TerminologyConstraintComposer(
            encode_fn=spy,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "serpent", "serpent")]
        plan = c.compose(
            source_text="In the beginning God created the heavens",
            glossary=entries,
            base_surfaces=frozenset({"lumière"}),
            base_ids=frozenset({1000}),
        )
        assert calls == [], f"Encoder was called unexpectedly: {calls}"
        assert not plan.gbs_required

    def test_encoder_called_for_active_entries(self):
        """Encoder is called with each active entry's target phrase."""
        m = _import_mod()
        spy, calls = make_spy_encode()
        c = m.TerminologyConstraintComposer(
            encode_fn=spy,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        c.compose(
            source_text="In the beginning God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert "Dieu" in calls

    def test_encoder_called_once_per_distinct_phrase(self):
        """Two entries with same target phrase encode at least once."""
        m = _import_mod()
        spy, calls = make_spy_encode()
        c = m.TerminologyConstraintComposer(
            encode_fn=spy,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "god", "Dieu"),
        ]
        c.compose(
            source_text="God the deity",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert calls.count("Dieu") >= 1


# ---------------------------------------------------------------------------
# §4 — License union and provenance
# ---------------------------------------------------------------------------

class TestLicenseUnion:
    """Compose merges glossary-derived surfaces/IDs with base licenses."""

    def test_active_entry_ids_added_to_license(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # "Dieu" -> [68, 105, 101, 117] (D=68, i=105, e=101, u=117)
        # All 4 lexical IDs should appear in licensed_token_ids
        for ch in "Dieu":
            assert ord(ch) in plan.licensed_token_ids, (
                f"Missing token ID {ord(ch)} for char {ch!r}"
            )

    def test_active_entry_surfaces_added_to_license(self):
        """Surfaces are full lexical units from human phrase text, not per-token decoded chars.

        _extract_phrase_surfaces('Dieu') = frozenset({'Dieu'}) — the whole word,
        not individual characters.  This is the tokenizer-safe behavior.
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Surface is the whole word "Dieu", not individual chars
        assert "Dieu" in plan.licensed_surfaces, (
            "Surface 'Dieu' (whole word) must be in licensed_surfaces"
        )

    def test_multiword_phrase_surfaces_split_on_whitespace(self):
        """Multi-word phrase surfaces are each individual word token."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God created", "Dieu créa")]
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # "Dieu créa" splits into {"Dieu", "créa"}
        assert "Dieu" in plan.licensed_surfaces
        assert "créa" in plan.licensed_surfaces

    def test_base_surfaces_preserved_in_union(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        base = frozenset({"lumière", "cieux"})
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=base,
            base_ids=frozenset({9000}),
            backend_supports_grammar_gbs=True,
        )
        assert "lumière" in plan.licensed_surfaces
        assert "cieux" in plan.licensed_surfaces
        assert 9000 in plan.licensed_token_ids

    def test_base_inputs_not_mutated(self):
        """compose() must not mutate the base_surfaces or base_ids frozensets."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        base_s = frozenset({"lumière"})
        base_i = frozenset({9000})
        c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=base_s,
            base_ids=base_i,
            backend_supports_grammar_gbs=True,
        )
        # frozensets are immutable — check they are unchanged objects
        assert base_s == frozenset({"lumière"})
        assert base_i == frozenset({9000})

    def test_no_trigger_identity_surfaces(self):
        """No active entries → licensed_surfaces equals base_surfaces exactly."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "serpent", "serpent")]
        base_s = frozenset({"lumière", "cieux", "terre"})
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=entries,
            base_surfaces=base_s,
            base_ids=frozenset({1, 2, 3}),
        )
        assert plan.licensed_surfaces == base_s

    def test_no_trigger_identity_ids(self):
        """No active entries → licensed_token_ids equals base_ids exactly."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "serpent", "serpent")]
        base_i = frozenset({1, 2, 3})
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=entries,
            base_surfaces=frozenset({"a"}),
            base_ids=base_i,
        )
        assert plan.licensed_token_ids == base_i

    def test_no_trigger_gbs_required_false(self):
        """No active entries → gbs_required is False."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.gbs_required is False

    def test_no_trigger_forced_phrases_empty(self):
        """No active entries → forced_phrases is an empty tuple (not list)."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.forced_phrases == ()

    def test_no_trigger_active_entry_ids_empty(self):
        """No active entries → active_entry_ids is an empty tuple (not list)."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.active_entry_ids == ()

    def test_multiple_entries_dedup_ids_union(self):
        """Multiple active entries sharing a token produce deduplicated IDs."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Di"),   # D=68, i=105
            m.GlossaryEntry("G2", "heaven", "Di"),  # same phrase -> same IDs
        ]
        plan = c.compose(
            source_text="God created the heaven",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # IDs should appear exactly once
        assert plan.licensed_token_ids == frozenset({68, 105})

    def test_multiple_entries_provenance_preserved(self):
        """When two entries share a surface/ID, both entry IDs appear in provenance."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Di"),
            m.GlossaryEntry("G2", "heaven", "Di"),
        ]
        plan = c.compose(
            source_text="God created the heaven",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Surface 'Di' (whole word, from both entries) should trace to both G1 and G2
        assert "G1" in plan.surface_provenance.get("Di", frozenset())
        assert "G2" in plan.surface_provenance.get("Di", frozenset())
        # Token ID 68 ('D') should trace to both G1 and G2
        assert "G1" in plan.id_provenance.get(68, frozenset())
        assert "G2" in plan.id_provenance.get(68, frozenset())

    def test_single_entry_provenance_surface(self):
        """Single active entry: the whole-word surface maps to that entry's ID."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Surface unit is "Dieu" (whole word), not individual chars
        assert "G1" in plan.surface_provenance.get("Dieu", frozenset()), (
            "Surface 'Dieu' (whole word) not traced to G1"
        )

    def test_single_entry_provenance_ids(self):
        """Single active entry: each lexical token ID maps to that entry's ID."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        for ch in "Dieu":
            assert "G1" in plan.id_provenance.get(ord(ch), frozenset()), (
                f"Token ID {ord(ch)} not traced to G1"
            )

    def test_space_surface_not_added_to_licensed_surfaces(self):
        """Space surface ' ' must NOT be in licensed_surfaces.

        Surfaces come from human-text phrase splitting on whitespace — space is
        the delimiter, not a surface unit.  This holds regardless of whether the
        encoder produces a space token ID.
        """
        m = _import_mod()

        def multi_word_encode(text: str) -> list[int]:
            # Returns char IDs including space char between words
            return [ord(c) for c in text]

        c = m.TerminologyConstraintComposer(
            encode_fn=multi_word_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "A B")]  # A=65, space=32, B=66
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert " " not in plan.licensed_surfaces, (
            "Space surface ' ' must not be in licensed_surfaces"
        )

    def test_space_token_id_included_in_licensed_ids_new_behavior(self):
        """Space token ID IS included in licensed_token_ids (tokenizer-safe behavior).

        Under the old per-ID-decode approach, space tokens were silently dropped
        because decode(space_id) == ' ' (whitespace).  This was the BPE bug:
        any token whose individual decode happened to be whitespace was excluded,
        even if it encoded semantically meaningful content (e.g. SentencePiece
        ▁-prefix tokens that decode to a leading space).

        Under the new design, ALL encoded IDs are licensed unless an EXPLICIT
        is_layout_id_fn predicate excludes them.  The space token ID (32 for ' ')
        is now correctly included.
        """
        m = _import_mod()
        SPACE_ID = 32  # ord(' ')

        def multi_word_encode(text: str) -> list[int]:
            return [ord(c) for c in text]

        c = m.TerminologyConstraintComposer(
            encode_fn=multi_word_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "A B")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # NEW BEHAVIOR: space ID IS in licensed_token_ids (no silent whitespace drop)
        assert SPACE_ID in plan.licensed_token_ids, (
            "Space token ID 32 MUST be in licensed_token_ids (tokenizer-safe: "
            "no silent per-ID whitespace dropping)"
        )

    def test_is_layout_id_fn_excludes_ids(self):
        """is_layout_id_fn predicate explicitly excludes layout/structural IDs."""
        m = _import_mod()
        SPACE_ID = 32

        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
            is_layout_id_fn=lambda tid: tid == SPACE_ID,  # only exclude space explicitly
        )
        entries = [m.GlossaryEntry("G1", "God", "A B")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # SPACE_ID explicitly excluded by is_layout_id_fn
        assert SPACE_ID not in plan.licensed_token_ids, (
            "Space token ID 32 was explicitly excluded by is_layout_id_fn"
        )
        # But non-space IDs (A=65, B=66) must still be present
        assert 65 in plan.licensed_token_ids
        assert 66 in plan.licensed_token_ids

    def test_active_entries_gbs_required_true(self):
        """Active entries → gbs_required is True."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_active_entries_forced_phrases_contains_target(self):
        """Active entries → forced_phrases contains each target phrase."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert "Dieu" in plan.forced_phrases

    def test_active_entries_active_entry_ids(self):
        """Active entries → active_entry_ids lists their IDs."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
        ]
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert "G1" in plan.active_entry_ids
        assert "G2" in plan.active_entry_ids


# ---------------------------------------------------------------------------
# §5 — Capability pre-flight
# ---------------------------------------------------------------------------

class TestCapabilityPreflight:
    def test_active_gbs_backend_missing_capability_raises(self):
        """Active entries + backend lacks grammar+GBS capability → MissingCapabilityError."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.MissingCapabilityError):
            c.compose(
                source_text="God created the heavens",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=False,  # explicit: backend lacks it
            )

    def test_active_gbs_default_backend_flag_raises(self):
        """Active entries with default backend flag (False) → MissingCapabilityError."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.MissingCapabilityError):
            c.compose(
                source_text="God created the heavens",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                # backend_supports_grammar_gbs not passed → default False
            )

    def test_missing_capability_error_is_typed(self):
        """MissingCapabilityError is a distinct typed error, not ValueError."""
        m = _import_mod()
        assert issubclass(m.MissingCapabilityError, Exception)

    def test_no_active_entries_no_capability_check(self):
        """No active entries → capability check irrelevant, no error raised."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "serpent", "serpent")]
        # Should not raise even with backend_supports_grammar_gbs=False (the default)
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=entries,
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.gbs_required is False

    def test_missing_capability_error_message_informative(self):
        """MissingCapabilityError message must mention grammar and GBS."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.MissingCapabilityError) as exc_info:
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
            )
        msg = str(exc_info.value).lower()
        assert "grammar" in msg or "gbs" in msg or "composition" in msg


# ---------------------------------------------------------------------------
# §6 — Reachability pre-flight
# ---------------------------------------------------------------------------

class TestReachabilityPreflight:
    """Reachability checks on forced phrases."""

    def test_empty_encoding_raises_unreachable_phrase_error(self):
        """A phrase that encodes to empty token list raises UnreachablePhraseError."""
        m = _import_mod()

        def empty_encode(text: str) -> list[int]:
            return []

        c = m.TerminologyConstraintComposer(
            encode_fn=empty_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.UnreachablePhraseError) as exc_info:
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=True,
            )
        assert exc_info.value.phrase == "Dieu"

    def test_roundtrip_mismatch_raises_unreachable_phrase_error(self):
        """decode_sequence_fn returning different text than the phrase raises error."""
        m = _import_mod()

        def garbling_encode(text: str) -> list[int]:
            return [1, 2, 3]  # some IDs

        def garbling_decode_sequence(ids: list[int]) -> str:
            return "COMPLETELY DIFFERENT"  # round-trip fails

        c = m.TerminologyConstraintComposer(
            encode_fn=garbling_encode,
            decode_sequence_fn=garbling_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.UnreachablePhraseError) as exc_info:
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=True,
            )
        assert exc_info.value.phrase == "Dieu"
        assert "round-trip" in exc_info.value.reason.lower() or "mismatch" in exc_info.value.reason.lower()

    def test_roundtrip_whitespace_normalized_passes(self):
        """Round-trip comparison is NFKC + whitespace-strip tolerant."""
        m = _import_mod()

        def padded_decode_sequence(ids: list[int]) -> str:
            # Returns phrase with leading/trailing whitespace — still passes
            return "  " + "".join(chr(i) for i in ids) + "  "

        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=padded_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        # Should NOT raise — strip normalizes the whitespace difference
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_unreachable_phrase_error_is_typed(self):
        """UnreachablePhraseError is a distinct typed error."""
        m = _import_mod()
        assert issubclass(m.UnreachablePhraseError, Exception)

    def test_unreachable_phrase_has_phrase_attribute(self):
        """UnreachablePhraseError carries .phrase attribute."""
        m = _import_mod()

        def empty_encode(text: str) -> list[int]:
            return []

        c = m.TerminologyConstraintComposer(
            encode_fn=empty_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.UnreachablePhraseError) as exc_info:
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=True,
            )
        assert exc_info.value.phrase == "Dieu"

    def test_reachability_callback_false_raises(self):
        """A reachability callback returning False → UnreachablePhraseError.

        The callback receives a PhraseReachabilityRequest (not a bare string).
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]

        def always_false_reachability(req) -> bool:
            return False

        with pytest.raises(m.UnreachablePhraseError):
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=True,
                grammar_reachability_fn=always_false_reachability,
            )

    def test_reachability_callback_true_does_not_raise(self):
        """A reachability callback returning True → no error."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]

        def always_true_reachability(req) -> bool:
            return True

        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
            grammar_reachability_fn=always_true_reachability,
        )
        assert plan.gbs_required is True

    def test_reachability_callback_receives_request_not_string(self):
        """Reachability callback receives a PhraseReachabilityRequest, not a string."""
        m = _import_mod()
        received: list = []

        def spy_reachability(req) -> bool:
            received.append(req)
            return True

        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
        ]
        c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
            grammar_reachability_fn=spy_reachability,
        )
        assert len(received) == 2
        # Each item must be a PhraseReachabilityRequest (not a string)
        for req in received:
            assert isinstance(req, m.PhraseReachabilityRequest), (
                f"Expected PhraseReachabilityRequest, got {type(req)}"
            )

    def test_reachability_callback_receives_phrase(self):
        """Reachability callback PhraseReachabilityRequest.phrase holds the target phrase."""
        m = _import_mod()
        received_phrases: list[str] = []

        def spy_reachability(req) -> bool:
            received_phrases.append(req.phrase)
            return True

        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
        ]
        c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
            grammar_reachability_fn=spy_reachability,
        )
        assert "Dieu" in received_phrases
        assert "cieux" in received_phrases

    def test_no_reachability_callback_no_error_by_default(self):
        """Without a callback, reachability check skips the callback step."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        # Should succeed without callback
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_lexical_ids_not_in_combined_license_raises(self):
        """A lexical token ID not in combined license → UnreachablePhraseError.

        This test verifies the correct behaviour: IDs from the phrase are added
        to the combined license, so they ARE reachable.
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # All Dieu's char IDs must be in the combined license
        for ch in "Dieu":
            assert ord(ch) in plan.licensed_token_ids

    def test_unreachable_phrase_error_has_reason_attribute(self):
        """UnreachablePhraseError carries a non-empty .reason attribute."""
        m = _import_mod()

        def empty_encode(text: str) -> list[int]:
            return []

        c = m.TerminologyConstraintComposer(
            encode_fn=empty_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        with pytest.raises(m.UnreachablePhraseError) as exc_info:
            c.compose(
                source_text="God created",
                glossary=entries,
                base_surfaces=frozenset(),
                base_ids=frozenset(),
                backend_supports_grammar_gbs=True,
            )
        assert exc_info.value.reason
        assert isinstance(exc_info.value.reason, str)


# ---------------------------------------------------------------------------
# §7 — ConstraintPlan structure and determinism
# ---------------------------------------------------------------------------

class TestConstraintPlanStructure:
    def test_plan_has_required_fields(self):
        """ConstraintPlan has all required fields."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
            backend_supports_grammar_gbs=True,
        )
        assert hasattr(plan, "licensed_surfaces")
        assert hasattr(plan, "licensed_token_ids")
        assert hasattr(plan, "forced_phrases")
        assert hasattr(plan, "active_entry_ids")
        assert hasattr(plan, "gbs_required")
        assert hasattr(plan, "surface_provenance")
        assert hasattr(plan, "id_provenance")

    def test_licensed_surfaces_is_frozenset(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.licensed_surfaces, frozenset)

    def test_licensed_token_ids_is_frozenset(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.licensed_token_ids, frozenset)

    def test_forced_phrases_is_tuple(self):
        """forced_phrases is a tuple (not list) — ConstraintPlan is frozen."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.forced_phrases, tuple)

    def test_active_entry_ids_is_tuple(self):
        """active_entry_ids is a tuple (not list) — ConstraintPlan is frozen."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.active_entry_ids, tuple)

    def test_constraint_plan_is_frozen(self):
        """ConstraintPlan is a frozen dataclass — attribute assignment must raise."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            plan.gbs_required = False  # type: ignore[misc]

    def test_surface_provenance_is_mapping_proxy(self):
        """surface_provenance is a read-only MappingProxyType."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.surface_provenance, types.MappingProxyType)
        with pytest.raises(TypeError):
            plan.surface_provenance["new_key"] = frozenset()  # type: ignore[index]

    def test_id_provenance_is_mapping_proxy(self):
        """id_provenance is a read-only MappingProxyType."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.id_provenance, types.MappingProxyType)
        with pytest.raises(TypeError):
            plan.id_provenance[9999] = frozenset()  # type: ignore[index]

    def test_compose_is_deterministic(self):
        """Calling compose twice with same inputs produces identical plans."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "heavens", "cieux"),
        ]
        kwargs = dict(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset({"lumière"}),
            base_ids=frozenset({100}),
            backend_supports_grammar_gbs=True,
        )
        p1 = c.compose(**kwargs)
        p2 = c.compose(**kwargs)
        assert p1.licensed_surfaces == p2.licensed_surfaces
        assert p1.licensed_token_ids == p2.licensed_token_ids
        assert p1.forced_phrases == p2.forced_phrases
        assert p1.active_entry_ids == p2.active_entry_ids
        assert p1.gbs_required == p2.gbs_required

    def test_compose_forced_phrases_deduplicated(self):
        """Two entries with same target_phrase produce one forced phrase entry."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entries = [
            m.GlossaryEntry("G1", "God", "Dieu"),
            m.GlossaryEntry("G2", "god", "Dieu"),  # same phrase
        ]
        plan = c.compose(
            source_text="God the deity",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.forced_phrases.count("Dieu") == 1

    def test_provenance_dicts_have_frozenset_values(self):
        """surface_provenance and id_provenance values are frozensets of entry_id strings."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        for v in plan.surface_provenance.values():
            assert isinstance(v, frozenset)
        for v in plan.id_provenance.values():
            assert isinstance(v, frozenset)


# ---------------------------------------------------------------------------
# §8 — GBS documentation / design-boundary assertions
# ---------------------------------------------------------------------------

class TestGBSDocumentationBoundary:
    def test_gbs_required_false_no_backend_constraint(self):
        """gbs_required=False means backend GBS capability is irrelevant (no check)."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        # No active entries → gbs_required=False; backend_supports_grammar_gbs=False is OK
        plan = c.compose(
            source_text="In the beginning",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
            backend_supports_grammar_gbs=False,
        )
        assert plan.gbs_required is False

    def test_constraint_plan_docstring_mentions_gbs_inclusion(self):
        """ConstraintPlan class has a docstring mentioning inclusion guarantee."""
        m = _import_mod()
        doc = m.ConstraintPlan.__doc__ or ""
        assert "inclusion" in doc.lower() or "guarantee" in doc.lower()


# ---------------------------------------------------------------------------
# §9 — PhraseReachabilityRequest rich frozen object
# ---------------------------------------------------------------------------

class TestPhraseReachabilityRequest:
    """Verify the PhraseReachabilityRequest passed to the callback is correct."""

    def _capture_request(self, m, phrase: str, allowed_sep=frozenset(), allowed_punc=frozenset()):
        """Helper: run compose with a spy callback and return the captured request."""
        captured: list = []

        def spy(req) -> bool:
            captured.append(req)
            return True

        c = m.TerminologyConstraintComposer(
            encode_fn=char_encode,
            decode_sequence_fn=char_decode_sequence,
        )
        entry_id = "G_test"
        entries = [m.GlossaryEntry(entry_id, "God", phrase)]
        c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset({"base_surf"}),
            base_ids=frozenset({9999}),
            backend_supports_grammar_gbs=True,
            grammar_reachability_fn=spy,
            allowed_separators=allowed_sep,
            allowed_punctuation=allowed_punc,
        )
        assert len(captured) == 1
        return captured[0]

    def test_request_phrase_attribute(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert req.phrase == "Dieu"

    def test_request_encoded_ids_attribute(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert req.encoded_ids == tuple(char_encode("Dieu"))

    def test_request_encoded_ids_is_tuple(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert isinstance(req.encoded_ids, tuple)

    def test_request_phrase_surfaces_attribute(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        # _extract_phrase_surfaces("Dieu") = frozenset({"Dieu"})
        assert isinstance(req.phrase_surfaces, frozenset)
        assert "Dieu" in req.phrase_surfaces

    def test_request_licensed_surfaces_contains_base_and_glossary(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert "base_surf" in req.licensed_surfaces
        assert "Dieu" in req.licensed_surfaces

    def test_request_licensed_token_ids_contains_base_and_glossary(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert 9999 in req.licensed_token_ids
        for ch in "Dieu":
            assert ord(ch) in req.licensed_token_ids

    def test_request_allowed_separators_forwarded(self):
        m = _import_mod()
        seps = frozenset({"-", " "})
        req = self._capture_request(m, "Dieu", allowed_sep=seps)
        assert req.allowed_separators == seps

    def test_request_allowed_punctuation_forwarded(self):
        m = _import_mod()
        punc = frozenset({".", ",", "!"})
        req = self._capture_request(m, "Dieu", allowed_punc=punc)
        assert req.allowed_punctuation == punc

    def test_request_is_frozen(self):
        """PhraseReachabilityRequest is a frozen dataclass."""
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        with pytest.raises((AttributeError, TypeError)):
            req.phrase = "tampered"  # type: ignore[misc]

    def test_request_default_separators_empty(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert req.allowed_separators == frozenset()

    def test_request_default_punctuation_empty(self):
        m = _import_mod()
        req = self._capture_request(m, "Dieu")
        assert req.allowed_punctuation == frozenset()


# ---------------------------------------------------------------------------
# §10 — BPE / SentencePiece / byte-level tokenizer safety (RED regressions)
# ---------------------------------------------------------------------------

class TestBPETokenizerSafety:
    """Regression tests for tokenizer-safety bugs present in the old per-ID-decode design.

    These tests document the correct behavior of the new full-sequence decode API
    when faced with realistic BPE, SentencePiece, and byte-level tokenizers.

    RED REGRESSION NOTES
    --------------------
    Old design (per-ID decode):
    - Decoded each token ID individually: decode_fn(id) → str
    - Silently dropped IDs whose individual decode was whitespace
    - Polluted surfaces with tokenizer artifacts (▁ markers, incomplete byte seqs)

    New design (full-sequence decode + human-text surfaces):
    - decode_sequence_fn(ids) → str for round-trip verification only
    - Surfaces come from human phrase text (not tokenizer decode)
    - All IDs licensed unless explicit is_layout_id_fn excludes them
    - BPE fragment/prefix tokens never silently dropped
    """

    # --- Fake BPE tokenizer with fragment/prefix tokens ---

    # A fragment ID whose individual decode would return "" (empty) under old approach
    BPE_FRAGMENT_ID = 9001

    @staticmethod
    def bpe_encode_with_fragment(text: str) -> list[int]:
        """Fake BPE: inserts a fragment prefix token before normal chars."""
        if text in ("Dieu", "lumière"):
            # BPE splits: fragment prefix + char IDs
            return [TestBPETokenizerSafety.BPE_FRAGMENT_ID] + [ord(c) for c in text]
        return [ord(c) for c in text]

    @staticmethod
    def bpe_decode_sequence_correct(ids: list[int]) -> str:
        """Full-sequence decode: the fragment is absorbed, rest decoded normally."""
        result = ""
        for tid in ids:
            if tid == TestBPETokenizerSafety.BPE_FRAGMENT_ID:
                pass  # absorbed BPE prefix marker
            else:
                result += chr(tid)
        return result

    def test_bpe_fragment_id_included_in_license(self):
        """BPE fragment token ID must be in licensed_token_ids (not silently dropped).

        RED REGRESSION: old per-ID-decode code called decode_fn(BPE_FRAGMENT_ID)
        and got "" (empty string), then silently excluded it from the ID license.
        This is wrong — a BPE fragment token may be essential for the grammar.
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.bpe_encode_with_fragment,
            decode_sequence_fn=self.bpe_decode_sequence_correct,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert self.BPE_FRAGMENT_ID in plan.licensed_token_ids, (
            "BPE fragment token ID must be in licensed_token_ids — "
            "old per-ID-decode code silently dropped it (BPE bug)"
        )

    def test_bpe_roundtrip_passes_with_fragment(self):
        """Round-trip verification passes when full-sequence decode handles fragments."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.bpe_encode_with_fragment,
            decode_sequence_fn=self.bpe_decode_sequence_correct,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        # Should not raise UnreachablePhraseError
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_bpe_surfaces_from_human_text_not_tokenizer(self):
        """Surfaces come from human phrase text, not tokenizer decode output.

        Even though the BPE encoding includes a fragment token that decodes to
        nothing or an artifact, the surface 'Dieu' comes from the human phrase
        text and is correct.
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.bpe_encode_with_fragment,
            decode_sequence_fn=self.bpe_decode_sequence_correct,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Human-text surface "Dieu" must be present
        assert "Dieu" in plan.licensed_surfaces
        # Tokenizer artifacts must NOT appear as surfaces
        assert "" not in plan.licensed_surfaces

    # --- SentencePiece-style leading-space tokens ---

    SP_DIEU_ID = 8001   # ▁Dieu token (SentencePiece prepends U+2581)

    @staticmethod
    def sp_encode(text: str) -> list[int]:
        """Fake SentencePiece: encodes 'Dieu' as a single ▁Dieu token."""
        if text == "Dieu":
            return [TestBPETokenizerSafety.SP_DIEU_ID]
        return [ord(c) for c in text]

    @staticmethod
    def sp_decode_sequence(ids: list[int]) -> str:
        """Full-sequence decode: strips ▁ prefix correctly."""
        if ids == [TestBPETokenizerSafety.SP_DIEU_ID]:
            return "Dieu"  # properly decoded, no ▁ artifact
        return "".join(chr(i) for i in ids)

    def test_sentencepiece_leading_space_roundtrip(self):
        """SentencePiece-style encode/decode: round-trip passes for 'Dieu'."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.sp_encode,
            decode_sequence_fn=self.sp_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert self.SP_DIEU_ID in plan.licensed_token_ids
        assert "Dieu" in plan.licensed_surfaces  # from human text, not tokenizer decode

    def test_sentencepiece_no_prefix_artifact_in_surfaces(self):
        """SentencePiece ▁ prefix must NOT pollute licensed_surfaces.

        Old per-ID-decode approach: decode_fn(SP_DIEU_ID) → '▁Dieu', which
        contains the ▁ marker.  That artifact could be added as a surface.
        New approach: surfaces come from human text 'Dieu' directly — no ▁.
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.sp_encode,
            decode_sequence_fn=self.sp_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # ▁Dieu (with prefix artifact) must NOT be a surface
        assert "\u2581Dieu" not in plan.licensed_surfaces
        assert "Dieu" in plan.licensed_surfaces

    # --- Multi-ID Unicode: single visible char split across multiple BPE tokens ---

    # Simulate "è" (U+00E8) encoded as two byte tokens [0xC3, 0xA8] in fake UTF-8
    BYTE_C3_ID = 7001  # Fake ID for first byte of è
    BYTE_A8_ID = 7002  # Fake ID for second byte of è

    @staticmethod
    def byte_encode_unicode(text: str) -> list[int]:
        """Fake byte-level encoder: 'è' → two byte tokens, others char-level."""
        if text == "lumière":
            # 'è' splits into two byte tokens, rest are char-level
            result = []
            for ch in "lumi":
                result.append(ord(ch))
            result.append(TestBPETokenizerSafety.BYTE_C3_ID)
            result.append(TestBPETokenizerSafety.BYTE_A8_ID)
            for ch in "re":
                result.append(ord(ch))
            return result
        return [ord(c) for c in text]

    @staticmethod
    def byte_decode_sequence_unicode(ids: list[int]) -> str:
        """Full-sequence decode: byte tokens for 'è' assembled correctly."""
        result = ""
        i = 0
        while i < len(ids):
            if ids[i] == TestBPETokenizerSafety.BYTE_C3_ID and i + 1 < len(ids) and ids[i + 1] == TestBPETokenizerSafety.BYTE_A8_ID:
                result += "è"
                i += 2
            else:
                result += chr(ids[i])
                i += 1
        return result

    def test_multi_id_unicode_roundtrip(self):
        """Multi-ID Unicode (è split into 2 byte tokens): round-trip passes."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.byte_encode_unicode,
            decode_sequence_fn=self.byte_decode_sequence_unicode,
        )
        entries = [m.GlossaryEntry("G1", "light", "lumière")]
        plan = c.compose(
            source_text="light",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_multi_id_unicode_both_byte_ids_licensed(self):
        """Both byte-level token IDs for 'è' must appear in licensed_token_ids.

        RED REGRESSION: old per-ID-decode code would attempt to decode each byte
        ID individually.  Byte 0xC3 alone is incomplete UTF-8, producing a
        mojibake or empty string.  The old code might silently drop it.
        New code: BOTH byte token IDs are included (no per-ID decode involved).
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.byte_encode_unicode,
            decode_sequence_fn=self.byte_decode_sequence_unicode,
        )
        entries = [m.GlossaryEntry("G1", "light", "lumière")]
        plan = c.compose(
            source_text="light",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert self.BYTE_C3_ID in plan.licensed_token_ids, (
            "First byte token of 'è' must be in licensed_token_ids"
        )
        assert self.BYTE_A8_ID in plan.licensed_token_ids, (
            "Second byte token of 'è' must be in licensed_token_ids"
        )

    def test_multi_id_unicode_surface_from_human_text(self):
        """Surface for lumière comes from human text, not from byte-ID decode output."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.byte_encode_unicode,
            decode_sequence_fn=self.byte_decode_sequence_unicode,
        )
        entries = [m.GlossaryEntry("G1", "light", "lumière")]
        plan = c.compose(
            source_text="light",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Human-text surface "lumière" must be present
        assert "lumière" in plan.licensed_surfaces

    # --- Per-ID empty decode but valid full decode ---

    EMPTY_DECODE_ID = 6001  # Token whose per-ID decode is "" but valid in sequence

    @staticmethod
    def empty_per_id_encode(text: str) -> list[int]:
        """Encoder that includes a token with per-ID empty decode."""
        if text == "Dieu":
            return [TestBPETokenizerSafety.EMPTY_DECODE_ID, ord('D'), ord('i'), ord('e'), ord('u')]
        return [ord(c) for c in text]

    @staticmethod
    def empty_per_id_decode_sequence(ids: list[int]) -> str:
        """Full-sequence decode: the empty-per-ID token is a silent marker."""
        result = ""
        for tid in ids:
            if tid == TestBPETokenizerSafety.EMPTY_DECODE_ID:
                pass  # silent marker, contributes nothing to output
            else:
                result += chr(tid)
        return result

    def test_empty_per_id_token_included_in_license(self):
        """Token whose per-ID decode is '' must still be in licensed_token_ids.

        RED REGRESSION: old code called decode_fn(id) for each token; if the
        result was '' (empty), the token ID was excluded from the lexical license.
        This is incorrect — a structural marker token may still be needed for
        the grammar to produce the correct output.
        New code: ALL encoded IDs are licensed (no per-ID decode exclusion).
        """
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.empty_per_id_encode,
            decode_sequence_fn=self.empty_per_id_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert self.EMPTY_DECODE_ID in plan.licensed_token_ids, (
            "Token with per-ID empty decode must still be in licensed_token_ids — "
            "old per-ID-decode code silently excluded it (BPE bug)"
        )

    def test_empty_per_id_roundtrip_passes(self):
        """Full-sequence roundtrip passes even when one token has empty per-ID decode."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.empty_per_id_encode,
            decode_sequence_fn=self.empty_per_id_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        # Should not raise — full-sequence decode gives "Dieu" which matches
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert plan.gbs_required is True

    def test_empty_per_id_id_provenance_preserved(self):
        """The empty-per-ID token ID's provenance is correctly tracked."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(
            encode_fn=self.empty_per_id_encode,
            decode_sequence_fn=self.empty_per_id_decode_sequence,
        )
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert "G1" in plan.id_provenance.get(self.EMPTY_DECODE_ID, frozenset()), (
            "Provenance for empty-per-ID token must include 'G1'"
        )
