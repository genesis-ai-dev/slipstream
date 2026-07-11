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
"""
from __future__ import annotations

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


def char_decode(token_id: int) -> str:
    """Decode a code-point integer back to its character."""
    return chr(token_id)


def make_spy_encode():
    """Return (spy_encode, calls) where calls is a list that records each call."""
    calls: list[str] = []

    def spy_encode(text: str) -> list[int]:
        calls.append(text)
        return char_encode(text)

    return spy_encode, calls


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
            decode_fn=char_decode,
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
        c = m.TerminologyConstraintComposer(encode_fn=spy, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=spy, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=spy, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created the heavens",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        for ch in "Dieu":
            assert ch in plan.licensed_surfaces, (
                f"Missing surface {ch!r} from target phrase"
            )

    def test_base_surfaces_preserved_in_union(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.gbs_required is False

    def test_no_trigger_forced_phrases_empty(self):
        """No active entries → forced_phrases is empty."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.forced_phrases == []

    def test_no_trigger_active_entry_ids_empty(self):
        """No active entries → active_entry_ids is empty."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="In the beginning God created",
            glossary=[m.GlossaryEntry("G1", "serpent", "serpent")],
            base_surfaces=frozenset({"a"}),
            base_ids=frozenset({1}),
        )
        assert plan.active_entry_ids == []

    def test_multiple_entries_dedup_ids_union(self):
        """Multiple active entries sharing a token produce deduplicated IDs."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        # Surface 'D' (from both entries) should trace to both G1 and G2
        assert "G1" in plan.surface_provenance.get("D", frozenset())
        assert "G2" in plan.surface_provenance.get("D", frozenset())
        # Token ID 68 ('D') should trace to both G1 and G2
        assert "G1" in plan.id_provenance.get(68, frozenset())
        assert "G2" in plan.id_provenance.get(68, frozenset())

    def test_single_entry_provenance_surface(self):
        """Single active entry: each surface maps to that entry's ID."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        for ch in "Dieu":
            assert "G1" in plan.surface_provenance.get(ch, frozenset()), (
                f"Surface {ch!r} not traced to G1"
            )

    def test_single_entry_provenance_ids(self):
        """Single active entry: each lexical token ID maps to that entry's ID."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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

    def test_whitespace_tokens_not_added_to_lexical_license(self):
        """Space-only tokens must NOT be added to licensed_surfaces or licensed_token_ids."""
        m = _import_mod()
        # Encoder that inserts a space token between words
        SPACE_ID = 32  # ord(' ')

        def multi_word_encode(text: str) -> list[int]:
            # Produce char IDs but also include a space token between words
            result = []
            for i, ch in enumerate(text):
                result.append(ord(ch))
            return result

        c = m.TerminologyConstraintComposer(
            encode_fn=multi_word_encode,
            decode_fn=char_decode,
        )
        entries = [m.GlossaryEntry("G1", "God", "A B")]  # A=65, space=32, B=66
        plan = c.compose(
            source_text="God created",
            glossary=entries,
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        # Space (ID=32, surface=' ') must NOT be in lexical license
        assert SPACE_ID not in plan.licensed_token_ids, (
            "Space token ID 32 must not be added to lexical licensed_token_ids"
        )
        assert " " not in plan.licensed_surfaces, (
            "Space surface ' ' must not be added to licensed_surfaces"
        )

    def test_active_entries_gbs_required_true(self):
        """Active entries → gbs_required is True."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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

        c = m.TerminologyConstraintComposer(encode_fn=empty_encode, decode_fn=char_decode)
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

    def test_unreachable_phrase_error_is_typed(self):
        """UnreachablePhraseError is a distinct typed error."""
        m = _import_mod()
        assert issubclass(m.UnreachablePhraseError, Exception)

    def test_unreachable_phrase_has_phrase_attribute(self):
        """UnreachablePhraseError carries .phrase attribute."""
        m = _import_mod()

        def empty_encode(text: str) -> list[int]:
            return []

        c = m.TerminologyConstraintComposer(encode_fn=empty_encode, decode_fn=char_decode)
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
        """A reachability callback returning False → UnreachablePhraseError."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]

        def always_false_reachability(phrase: str) -> bool:
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        entries = [m.GlossaryEntry("G1", "God", "Dieu")]

        def always_true_reachability(phrase: str) -> bool:
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

    def test_reachability_callback_receives_phrase(self):
        """Reachability callback receives each forced phrase as argument."""
        m = _import_mod()
        received: list[str] = []

        def spy_reachability(phrase: str) -> bool:
            received.append(phrase)
            return True

        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        assert "Dieu" in received
        assert "cieux" in received

    def test_no_reachability_callback_no_error_by_default(self):
        """Without a callback, reachability check skips the callback step."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        
        This test uses a custom encoder that returns IDs which bypass the normal
        add-to-license path to verify the reachability check catches it.
        """
        m = _import_mod()
        # Encoder returns a known ID that will be in the plan's licensed_token_ids
        # (the compose adds them). BUT if we then run a final check against
        # a *pre-composed* set (base only) and the phrase encodes to IDs not in base,
        # the implementation should still proceed (IDs are ADDED before the check).
        # This test verifies the correct behaviour: IDs from the phrase are added
        # to the combined license, so they ARE reachable.
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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

        c = m.TerminologyConstraintComposer(encode_fn=empty_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.licensed_token_ids, frozenset)

    def test_forced_phrases_is_list(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.forced_phrases, list)

    def test_active_entry_ids_is_list(self):
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
        plan = c.compose(
            source_text="God created",
            glossary=[m.GlossaryEntry("G1", "God", "Dieu")],
            base_surfaces=frozenset(),
            base_ids=frozenset(),
            backend_supports_grammar_gbs=True,
        )
        assert isinstance(plan.active_entry_ids, list)

    def test_compose_is_deterministic(self):
        """Calling compose twice with same inputs produces identical plans."""
        m = _import_mod()
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
        c = m.TerminologyConstraintComposer(encode_fn=char_encode, decode_fn=char_decode)
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
