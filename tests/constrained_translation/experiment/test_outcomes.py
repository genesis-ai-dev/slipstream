"""tests/constrained_translation/experiment/test_outcomes.py

Strict TDD tests for the safety and constraint-cost outcome taxonomy (Task 4).
Written BEFORE the implementation (RED phase).

Outcome taxonomy (mutually exclusive primary outcomes):
1. SAFE_SOURCE_ABSTENTION   — deterministic pre-generation unsupported source; no backend call
2. FALSE_SOURCE_ABSTENTION  — detector wrong: canonical unit IS in translated source (defect)
3. ACCEPTED_LICENSED        — source supported, generation passed all audits, no model UNK
4. CONSTRAINT_ARTIFACT      — surface licensed but rejected by grammar/tokenizer/machinery
5. EMITTED_HALLUCINATION    — model emitted unlicensed token ID or unattested surface
6. ESCAPED_HALLUCINATION    — emitted hallucination that would be accepted (invariant violation)
7. MODEL_GENERATED_UNK      — model generated [UNK:...] marker (must not)
8. OTHER_HARD_FAILURE       — backend/grammar/request failure not fitting above

Testing conventions:
- One behaviour per test function.
- Helpers are module-level to avoid fixture overhead.
- Class groupings reflect taxonomy sections.
"""
from __future__ import annotations

import pytest

from constrained_translation.experiment.outcomes import (
    Outcome,
    TranslationEvidence,
    EscapedHallucinationError,
    classify,
    guard_no_escaped_hallucination,
    OutcomeRollup,
    compute_rollup,
)


# ---------------------------------------------------------------------------
# Helpers — canonical evidence builders for each outcome
# ---------------------------------------------------------------------------

def _base_evidence(**overrides) -> TranslationEvidence:
    """Return a fully-specified ACCEPTED_LICENSED evidence, override fields as needed."""
    defaults = dict(
        item_id="GEN 1:1",
        source_text="In the beginning",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="Au commencement",
        exact_id_licensed=True,
        visible_surface_licensed=True,
        full_decode_match=True,
        control_token_valid=True,
        grammar_accepted=True,
        audit_passed=True,
        audit_reasons=[],
        model_generated_unk=False,
        surfaced_failure_artifact=None,
        final_translation="Au commencement",
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _safe_abstention_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:1",
        source_text="untranslatable_xyz",
        source_supported=False,
        false_abstention_detected=False,
        backend_called=False,
        raw_generation=None,
        exact_id_licensed=False,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=False,
        grammar_accepted=False,
        audit_passed=False,
        audit_reasons=[],
        model_generated_unk=False,
        surfaced_failure_artifact=None,
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _false_abstention_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:2",
        source_text="the earth",
        source_supported=False,
        false_abstention_detected=True,
        backend_called=False,
        raw_generation=None,
        exact_id_licensed=False,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=False,
        grammar_accepted=False,
        audit_passed=False,
        audit_reasons=[],
        model_generated_unk=False,
        surfaced_failure_artifact=None,
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _constraint_artifact_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:3",
        source_text="And God said",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="Et Dieu dit\x1b[0m",  # terminal-control byte in output
        exact_id_licensed=True,
        visible_surface_licensed=True,
        full_decode_match=True,
        control_token_valid=False,   # machinery rejection
        grammar_accepted=False,       # grammar rejected
        audit_passed=False,
        audit_reasons=["terminal control at non-final position"],
        model_generated_unk=False,
        surfaced_failure_artifact="Et Dieu dit\x1b[0m",
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _emitted_hallucination_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:4",
        source_text="And God saw",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="Et Dieu vit UNLICENSED",
        exact_id_licensed=False,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=True,
        grammar_accepted=True,
        audit_passed=False,
        audit_reasons=["token_id=999 decoded='UNLICENSED' not in token licence"],
        model_generated_unk=False,
        surfaced_failure_artifact="Et Dieu vit UNLICENSED",
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _escaped_hallucination_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:5",
        source_text="And called the light",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="Et appela UNLICENSED",
        exact_id_licensed=False,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=True,
        grammar_accepted=True,
        audit_passed=False,
        audit_reasons=["token_id=999 decoded='UNLICENSED' not in token licence"],
        model_generated_unk=False,
        surfaced_failure_artifact="Et appela UNLICENSED",
        final_translation="Et appela UNLICENSED",  # hallucination escaped to output
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _model_unk_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:6",
        source_text="And God said let",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="Et Dieu dit [UNK:let]",
        exact_id_licensed=True,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=True,
        grammar_accepted=True,
        audit_passed=False,
        audit_reasons=["token_id=77 decoded='[UNK:let]' is a [UNK:…] marker in model output"],
        model_generated_unk=True,
        surfaced_failure_artifact="Et Dieu dit [UNK:let]",
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


def _other_failure_evidence(**overrides) -> TranslationEvidence:
    defaults = dict(
        item_id="GEN 1:7",
        source_text="And God made",
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation=None,
        exact_id_licensed=False,
        visible_surface_licensed=False,
        full_decode_match=False,
        control_token_valid=False,
        grammar_accepted=False,
        audit_passed=False,
        audit_reasons=["backend timeout: connection refused"],
        model_generated_unk=False,
        surfaced_failure_artifact=None,
        final_translation=None,
    )
    defaults.update(overrides)
    return TranslationEvidence(**defaults)


# ===========================================================================
# §1 — TranslationEvidence dataclass API
# ===========================================================================

class TestTranslationEvidenceDataclass:
    def test_has_item_id_field(self):
        ev = _base_evidence()
        assert ev.item_id == "GEN 1:1"

    def test_has_source_text_field(self):
        ev = _base_evidence()
        assert ev.source_text == "In the beginning"

    def test_has_source_supported_field(self):
        ev = _base_evidence()
        assert ev.source_supported is True

    def test_has_false_abstention_detected_field(self):
        ev = _base_evidence()
        assert ev.false_abstention_detected is False

    def test_has_backend_called_field(self):
        ev = _base_evidence()
        assert ev.backend_called is True

    def test_has_raw_generation_field(self):
        ev = _base_evidence()
        assert ev.raw_generation == "Au commencement"

    def test_raw_generation_none_when_no_backend(self):
        ev = _safe_abstention_evidence()
        assert ev.raw_generation is None

    def test_has_exact_id_licensed_field(self):
        ev = _base_evidence()
        assert ev.exact_id_licensed is True

    def test_has_visible_surface_licensed_field(self):
        ev = _base_evidence()
        assert ev.visible_surface_licensed is True

    def test_has_full_decode_match_field(self):
        ev = _base_evidence()
        assert ev.full_decode_match is True

    def test_has_control_token_valid_field(self):
        ev = _base_evidence()
        assert ev.control_token_valid is True

    def test_has_grammar_accepted_field(self):
        ev = _base_evidence()
        assert ev.grammar_accepted is True

    def test_has_audit_passed_field(self):
        ev = _base_evidence()
        assert ev.audit_passed is True

    def test_has_audit_reasons_field(self):
        ev = _base_evidence()
        assert isinstance(ev.audit_reasons, list)
        assert ev.audit_reasons == []

    def test_has_model_generated_unk_field(self):
        ev = _base_evidence()
        assert ev.model_generated_unk is False

    def test_has_surfaced_failure_artifact_field(self):
        ev = _base_evidence()
        assert ev.surfaced_failure_artifact is None

    def test_has_final_translation_field(self):
        ev = _base_evidence()
        assert ev.final_translation == "Au commencement"

    def test_final_translation_none_for_rejected(self):
        ev = _emitted_hallucination_evidence()
        assert ev.final_translation is None

    def test_sentinel_surfaced_artifact_preserved_on_rejection(self):
        """surfaced_failure_artifact must hold the raw rejected text, not None."""
        ev = _constraint_artifact_evidence()
        assert ev.surfaced_failure_artifact is not None
        assert "Et Dieu dit" in ev.surfaced_failure_artifact

    def test_surfaced_artifact_distinct_from_final_translation(self):
        """Wrong-field guard: surfaced_failure_artifact ≠ final_translation sentinel."""
        ev = _emitted_hallucination_evidence()
        # surfaced_failure_artifact holds rejected text; final_translation is None
        assert ev.surfaced_failure_artifact == "Et Dieu vit UNLICENSED"
        assert ev.final_translation is None

    def test_accepted_not_inferred_from_nonempty_translation_string(self):
        """Classifier must read audit_passed, not len(final_translation) > 0."""
        # Build evidence with a non-empty final_translation but audit_passed=False
        # and hallucination present — must NOT classify as ACCEPTED_LICENSED.
        ev = _escaped_hallucination_evidence()
        assert ev.final_translation is not None and len(ev.final_translation) > 0
        outcome = classify(ev)
        assert outcome != Outcome.ACCEPTED_LICENSED


# ===========================================================================
# §2 — Outcome enum values
# ===========================================================================

class TestOutcomeEnum:
    def test_safe_source_abstention_exists(self):
        assert Outcome.SAFE_SOURCE_ABSTENTION is not None

    def test_false_source_abstention_exists(self):
        assert Outcome.FALSE_SOURCE_ABSTENTION is not None

    def test_accepted_licensed_exists(self):
        assert Outcome.ACCEPTED_LICENSED is not None

    def test_constraint_artifact_exists(self):
        assert Outcome.CONSTRAINT_ARTIFACT is not None

    def test_emitted_hallucination_exists(self):
        assert Outcome.EMITTED_HALLUCINATION is not None

    def test_escaped_hallucination_exists(self):
        assert Outcome.ESCAPED_HALLUCINATION is not None

    def test_model_generated_unk_exists(self):
        assert Outcome.MODEL_GENERATED_UNK is not None

    def test_other_hard_failure_exists(self):
        assert Outcome.OTHER_HARD_FAILURE is not None

    def test_all_eight_outcomes_distinct(self):
        outcomes = [
            Outcome.SAFE_SOURCE_ABSTENTION,
            Outcome.FALSE_SOURCE_ABSTENTION,
            Outcome.ACCEPTED_LICENSED,
            Outcome.CONSTRAINT_ARTIFACT,
            Outcome.EMITTED_HALLUCINATION,
            Outcome.ESCAPED_HALLUCINATION,
            Outcome.MODEL_GENERATED_UNK,
            Outcome.OTHER_HARD_FAILURE,
        ]
        assert len(set(outcomes)) == 8


# ===========================================================================
# §3 — classify: SAFE_SOURCE_ABSTENTION
# ===========================================================================

class TestClassifySafeSourceAbstention:
    def test_no_backend_call_unsupported_source_yields_safe_abstention(self):
        ev = _safe_abstention_evidence()
        assert classify(ev) == Outcome.SAFE_SOURCE_ABSTENTION

    def test_safe_abstention_requires_backend_not_called(self):
        """If backend WAS called, it cannot be safe abstention."""
        ev = _safe_abstention_evidence(backend_called=True)
        assert classify(ev) != Outcome.SAFE_SOURCE_ABSTENTION

    def test_safe_abstention_requires_source_not_supported(self):
        """If source IS supported, it cannot be safe abstention."""
        ev = _safe_abstention_evidence(source_supported=True)
        assert classify(ev) != Outcome.SAFE_SOURCE_ABSTENTION

    def test_safe_abstention_is_not_hallucination(self):
        ev = _safe_abstention_evidence()
        outcome = classify(ev)
        assert outcome not in (Outcome.EMITTED_HALLUCINATION, Outcome.ESCAPED_HALLUCINATION)

    def test_safe_abstention_raw_generation_is_none(self):
        """Safe abstentions produce no raw generation."""
        ev = _safe_abstention_evidence()
        assert ev.raw_generation is None

    def test_safe_abstention_final_translation_is_none(self):
        ev = _safe_abstention_evidence()
        assert ev.final_translation is None


# ===========================================================================
# §4 — classify: FALSE_SOURCE_ABSTENTION
# ===========================================================================

class TestClassifyFalseSourceAbstention:
    def test_false_abstention_detected_flag_triggers_outcome(self):
        ev = _false_abstention_evidence()
        assert classify(ev) == Outcome.FALSE_SOURCE_ABSTENTION

    def test_false_abstention_takes_priority_over_safe_abstention(self):
        """false_abstention_detected=True must dominate even when backend not called."""
        ev = _false_abstention_evidence(backend_called=False, source_supported=False)
        assert classify(ev) == Outcome.FALSE_SOURCE_ABSTENTION

    def test_false_abstention_distinct_from_safe_abstention(self):
        ev_safe = _safe_abstention_evidence()
        ev_false = _false_abstention_evidence()
        assert classify(ev_safe) != classify(ev_false)

    def test_safe_abstention_without_false_detection(self):
        """SAFE abstention only when false_abstention_detected=False."""
        ev = _safe_abstention_evidence(false_abstention_detected=False)
        assert classify(ev) == Outcome.SAFE_SOURCE_ABSTENTION

    def test_false_abstention_is_not_safe_abstention(self):
        ev = _false_abstention_evidence()
        assert classify(ev) != Outcome.SAFE_SOURCE_ABSTENTION


# ===========================================================================
# §5 — classify: ACCEPTED_LICENSED
# ===========================================================================

class TestClassifyAcceptedLicensed:
    def test_all_passing_evidence_yields_accepted_licensed(self):
        ev = _base_evidence()
        assert classify(ev) == Outcome.ACCEPTED_LICENSED

    def test_accepted_requires_backend_called(self):
        ev = _base_evidence(backend_called=False)
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_accepted_requires_audit_passed(self):
        ev = _base_evidence(audit_passed=False)
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_accepted_requires_no_model_unk(self):
        ev = _base_evidence(model_generated_unk=True)
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_accepted_requires_source_supported(self):
        ev = _base_evidence(source_supported=False, false_abstention_detected=False,
                            backend_called=True, audit_passed=True)
        # If source is not supported but backend was called and audit passed,
        # it should NOT be classified as ACCEPTED (inconsistent evidence → OTHER)
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_accepted_final_translation_not_none(self):
        """ACCEPTED_LICENSED units should have a final_translation."""
        ev = _base_evidence()
        assert ev.final_translation is not None

    def test_accepted_not_from_nonempty_string(self):
        """Classifier must read audit_passed, not infer from translation string."""
        # audit_passed=False but final_translation is non-empty (would be a bug/escaped)
        ev = _base_evidence(audit_passed=False, exact_id_licensed=False,
                            visible_surface_licensed=False,
                            final_translation="Et Dieu")
        outcome = classify(ev)
        assert outcome != Outcome.ACCEPTED_LICENSED


# ===========================================================================
# §6 — classify: CONSTRAINT_ARTIFACT
# ===========================================================================

class TestClassifyConstraintArtifact:
    def test_licensed_surface_machinery_rejection_yields_constraint_artifact(self):
        ev = _constraint_artifact_evidence()
        assert classify(ev) == Outcome.CONSTRAINT_ARTIFACT

    def test_constraint_artifact_requires_backend_called(self):
        ev = _constraint_artifact_evidence(backend_called=False)
        assert classify(ev) != Outcome.CONSTRAINT_ARTIFACT

    def test_constraint_artifact_requires_exact_id_licensed(self):
        """If IDs are unlicensed, it is hallucination, not constraint artifact."""
        ev = _constraint_artifact_evidence(exact_id_licensed=False)
        outcome = classify(ev)
        assert outcome in (Outcome.EMITTED_HALLUCINATION, Outcome.ESCAPED_HALLUCINATION)

    def test_constraint_artifact_requires_visible_surface_licensed(self):
        """If surface words are unlicensed, it is hallucination, not constraint artifact."""
        ev = _constraint_artifact_evidence(visible_surface_licensed=False)
        outcome = classify(ev)
        assert outcome in (Outcome.EMITTED_HALLUCINATION, Outcome.ESCAPED_HALLUCINATION)

    def test_constraint_artifact_requires_audit_not_passed(self):
        """If audit passed, it should be ACCEPTED_LICENSED, not CONSTRAINT_ARTIFACT."""
        ev = _constraint_artifact_evidence(audit_passed=True, control_token_valid=True,
                                           grammar_accepted=True,
                                           final_translation="Et Dieu dit")
        assert classify(ev) != Outcome.CONSTRAINT_ARTIFACT

    def test_constraint_artifact_does_not_swallow_unlicensed_word(self):
        """Must not classify as artifact if an actual unlicensed surface word is present."""
        ev = _constraint_artifact_evidence(
            visible_surface_licensed=False,
            exact_id_licensed=False,
            audit_reasons=["surface word 'BADWORD' is not in attested_vocab"]
        )
        outcome = classify(ev)
        assert outcome != Outcome.CONSTRAINT_ARTIFACT

    def test_constraint_artifact_raw_rejected_text_retained(self):
        """surfaced_failure_artifact must be present on constraint artifacts."""
        ev = _constraint_artifact_evidence()
        assert ev.surfaced_failure_artifact is not None

    def test_constraint_artifact_final_translation_none(self):
        """Rejected units have no final_translation."""
        ev = _constraint_artifact_evidence()
        assert ev.final_translation is None

    def test_grammar_rejection_is_constraint_artifact(self):
        """Grammar rejection with licensed surface is CONSTRAINT_ARTIFACT."""
        ev = _constraint_artifact_evidence(
            grammar_accepted=False,
            control_token_valid=True,
            audit_reasons=["grammar constraint rejected output"],
        )
        assert classify(ev) == Outcome.CONSTRAINT_ARTIFACT

    def test_control_token_rejection_is_constraint_artifact(self):
        """Terminal-control token at wrong position with licensed surface → artifact."""
        ev = _constraint_artifact_evidence(
            control_token_valid=False,
            grammar_accepted=True,
            audit_reasons=["terminal control at non-final position idx=3"],
        )
        assert classify(ev) == Outcome.CONSTRAINT_ARTIFACT


# ===========================================================================
# §7 — classify: EMITTED_HALLUCINATION
# ===========================================================================

class TestClassifyEmittedHallucination:
    def test_unlicensed_id_rejected_yields_emitted_hallucination(self):
        ev = _emitted_hallucination_evidence()
        assert classify(ev) == Outcome.EMITTED_HALLUCINATION

    def test_unlicensed_surface_yields_emitted_hallucination(self):
        ev = _emitted_hallucination_evidence(
            exact_id_licensed=True,  # IDs are licensed
            visible_surface_licensed=False,  # but surface recombination fails
            audit_reasons=["surface word 'novelword' is not in attested_vocab"],
        )
        assert classify(ev) == Outcome.EMITTED_HALLUCINATION

    def test_emitted_hallucination_requires_final_translation_none(self):
        """If final_translation is set, it is ESCAPED, not EMITTED."""
        ev = _emitted_hallucination_evidence(final_translation="Et Dieu vit UNLICENSED")
        assert classify(ev) == Outcome.ESCAPED_HALLUCINATION

    def test_emitted_hallucination_raw_rejected_text_retained(self):
        ev = _emitted_hallucination_evidence()
        assert ev.surfaced_failure_artifact == "Et Dieu vit UNLICENSED"

    def test_emitted_hallucination_is_rejected_failure(self):
        """EMITTED_HALLUCINATION must not be classified as success."""
        ev = _emitted_hallucination_evidence()
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_emitted_hallucination_distinct_from_constraint_artifact(self):
        ev_hall = _emitted_hallucination_evidence()
        ev_art = _constraint_artifact_evidence()
        assert classify(ev_hall) != classify(ev_art)


# ===========================================================================
# §8 — classify: ESCAPED_HALLUCINATION (invariant violation)
# ===========================================================================

class TestClassifyEscapedHallucination:
    def test_hallucination_with_nonempty_final_translation_is_escaped(self):
        ev = _escaped_hallucination_evidence()
        assert classify(ev) == Outcome.ESCAPED_HALLUCINATION

    def test_escaped_hallucination_requires_unlicensed_content(self):
        """If IDs and surface are licensed, it cannot be ESCAPED hallucination."""
        ev = _escaped_hallucination_evidence(
            exact_id_licensed=True,
            visible_surface_licensed=True,
            audit_passed=False,  # still fails, but due to machinery
            audit_reasons=["grammar constraint rejected output"],
        )
        assert classify(ev) != Outcome.ESCAPED_HALLUCINATION

    def test_escaped_hallucination_guard_raises_loudly(self):
        """guard_no_escaped_hallucination must raise EscapedHallucinationError."""
        ev = _escaped_hallucination_evidence()
        outcome = classify(ev)
        with pytest.raises(EscapedHallucinationError):
            guard_no_escaped_hallucination(outcome, ev)

    def test_guard_does_not_raise_for_accepted(self):
        ev = _base_evidence()
        outcome = classify(ev)
        guard_no_escaped_hallucination(outcome, ev)  # must not raise

    def test_guard_does_not_raise_for_emitted_hallucination(self):
        """EMITTED (rejected) hallucination does not trigger the escaped guard."""
        ev = _emitted_hallucination_evidence()
        outcome = classify(ev)
        guard_no_escaped_hallucination(outcome, ev)  # must not raise

    def test_guard_does_not_raise_for_safe_abstention(self):
        ev = _safe_abstention_evidence()
        outcome = classify(ev)
        guard_no_escaped_hallucination(outcome, ev)  # must not raise

    def test_escaped_hallucination_error_contains_item_id(self):
        ev = _escaped_hallucination_evidence()
        outcome = classify(ev)
        with pytest.raises(EscapedHallucinationError, match=ev.item_id):
            guard_no_escaped_hallucination(outcome, ev)

    def test_escaped_hallucination_classified_not_accepted(self):
        ev = _escaped_hallucination_evidence()
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_escaped_hallucination_classified_not_emitted(self):
        """ESCAPED is separate from EMITTED — distinct primary outcome."""
        ev = _escaped_hallucination_evidence()
        assert classify(ev) != Outcome.EMITTED_HALLUCINATION


# ===========================================================================
# §9 — classify: MODEL_GENERATED_UNK
# ===========================================================================

class TestClassifyModelGeneratedUnk:
    def test_model_unk_flag_yields_model_generated_unk(self):
        ev = _model_unk_evidence()
        assert classify(ev) == Outcome.MODEL_GENERATED_UNK

    def test_model_unk_requires_backend_called(self):
        ev = _model_unk_evidence(backend_called=False)
        assert classify(ev) != Outcome.MODEL_GENERATED_UNK

    def test_model_unk_is_separate_rejected_violation(self):
        """MODEL_GENERATED_UNK must not be ACCEPTED or CONSTRAINT_ARTIFACT."""
        ev = _model_unk_evidence()
        outcome = classify(ev)
        assert outcome not in (
            Outcome.ACCEPTED_LICENSED,
            Outcome.CONSTRAINT_ARTIFACT,
            Outcome.SAFE_SOURCE_ABSTENTION,
            Outcome.FALSE_SOURCE_ABSTENTION,
        )

    def test_model_unk_raw_text_retained(self):
        ev = _model_unk_evidence()
        assert "[UNK:let]" in ev.raw_generation

    def test_model_unk_surfaced_artifact_preserved(self):
        ev = _model_unk_evidence()
        assert ev.surfaced_failure_artifact is not None
        assert "[UNK:let]" in ev.surfaced_failure_artifact


# ===========================================================================
# §10 — classify: OTHER_HARD_FAILURE
# ===========================================================================

class TestClassifyOtherHardFailure:
    def test_backend_failure_not_fitting_other_categories_is_other(self):
        ev = _other_failure_evidence()
        assert classify(ev) == Outcome.OTHER_HARD_FAILURE

    def test_other_hard_failure_is_not_hallucination(self):
        ev = _other_failure_evidence()
        outcome = classify(ev)
        assert outcome not in (
            Outcome.EMITTED_HALLUCINATION,
            Outcome.ESCAPED_HALLUCINATION,
        )

    def test_other_hard_failure_is_not_accepted(self):
        ev = _other_failure_evidence()
        assert classify(ev) != Outcome.ACCEPTED_LICENSED


# ===========================================================================
# §11 — Mutual exclusivity
# ===========================================================================

class TestMutualExclusivity:
    """Verify that each canonical evidence maps to exactly one outcome."""

    _CANONICAL_EVIDENCES = [
        ("SAFE_SOURCE_ABSTENTION", _safe_abstention_evidence()),
        ("FALSE_SOURCE_ABSTENTION", _false_abstention_evidence()),
        ("ACCEPTED_LICENSED", _base_evidence()),
        ("CONSTRAINT_ARTIFACT", _constraint_artifact_evidence()),
        ("EMITTED_HALLUCINATION", _emitted_hallucination_evidence()),
        ("ESCAPED_HALLUCINATION", _escaped_hallucination_evidence()),
        ("MODEL_GENERATED_UNK", _model_unk_evidence()),
        ("OTHER_HARD_FAILURE", _other_failure_evidence()),
    ]

    def test_each_canonical_evidence_maps_to_expected_outcome(self):
        for name, ev in self._CANONICAL_EVIDENCES:
            result = classify(ev)
            expected = Outcome[name]
            assert result == expected, (
                f"Expected {name} for canonical evidence, got {result}"
            )

    def test_all_eight_canonical_outcomes_are_distinct(self):
        outcomes = [classify(ev) for _, ev in self._CANONICAL_EVIDENCES]
        assert len(set(outcomes)) == 8

    def test_no_single_evidence_maps_to_multiple_outcomes(self):
        """Classifier is a pure function — same input always same output."""
        for name, ev in self._CANONICAL_EVIDENCES:
            o1 = classify(ev)
            o2 = classify(ev)
            assert o1 == o2

    def test_false_abstention_dominates_safe_abstention(self):
        """When both flags could apply, FALSE_SOURCE_ABSTENTION wins."""
        ev = TranslationEvidence(
            item_id="X",
            source_text="text",
            source_supported=False,
            false_abstention_detected=True,
            backend_called=False,
            raw_generation=None,
            exact_id_licensed=False,
            visible_surface_licensed=False,
            full_decode_match=False,
            control_token_valid=False,
            grammar_accepted=False,
            audit_passed=False,
            audit_reasons=[],
            model_generated_unk=False,
            surfaced_failure_artifact=None,
            final_translation=None,
        )
        assert classify(ev) == Outcome.FALSE_SOURCE_ABSTENTION

    def test_escaped_hallucination_dominates_emitted(self):
        """ESCAPED must be returned, not EMITTED, when final_translation is set."""
        ev = _emitted_hallucination_evidence(final_translation="bad text")
        assert classify(ev) == Outcome.ESCAPED_HALLUCINATION

    def test_model_unk_dominates_constraint_artifact(self):
        """MODEL_GENERATED_UNK takes priority over CONSTRAINT_ARTIFACT."""
        ev = _constraint_artifact_evidence(
            model_generated_unk=True,
            audit_reasons=["model generated [UNK:...] marker"],
        )
        assert classify(ev) == Outcome.MODEL_GENERATED_UNK


# ===========================================================================
# §12 — Invariant: no unconstrained fallback or weakening
# ===========================================================================

class TestNoFallbackWeakening:
    def test_classifier_never_returns_none(self):
        for ev in [
            _safe_abstention_evidence(),
            _false_abstention_evidence(),
            _base_evidence(),
            _constraint_artifact_evidence(),
            _emitted_hallucination_evidence(),
            _escaped_hallucination_evidence(),
            _model_unk_evidence(),
            _other_failure_evidence(),
        ]:
            assert classify(ev) is not None

    def test_no_weakening_emitted_to_constraint_artifact(self):
        """Unlicensed visible word must never be relabelled as machinery failure."""
        ev = _emitted_hallucination_evidence()
        assert classify(ev) != Outcome.CONSTRAINT_ARTIFACT

    def test_no_weakening_escaped_to_accepted(self):
        """An escaped hallucination must never be relabelled as ACCEPTED_LICENSED."""
        ev = _escaped_hallucination_evidence()
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_no_weakening_model_unk_to_accepted(self):
        """Model-generated UNK must never be relabelled as ACCEPTED_LICENSED."""
        ev = _model_unk_evidence()
        assert classify(ev) != Outcome.ACCEPTED_LICENSED


# ===========================================================================
# §13 — OutcomeRollup dataclass
# ===========================================================================

class TestOutcomeRollupDataclass:
    def test_rollup_has_n_evaluated(self):
        rollup = compute_rollup([])
        assert hasattr(rollup, "n_evaluated")

    def test_rollup_has_n_generated(self):
        rollup = compute_rollup([])
        assert hasattr(rollup, "n_generated")

    def test_rollup_has_n_supported(self):
        rollup = compute_rollup([])
        assert hasattr(rollup, "n_supported")

    def test_rollup_has_all_eight_count_fields(self):
        rollup = compute_rollup([])
        assert hasattr(rollup, "n_safe_abstentions")
        assert hasattr(rollup, "n_false_abstentions")
        assert hasattr(rollup, "n_accepted")
        assert hasattr(rollup, "n_constraint_artifacts")
        assert hasattr(rollup, "n_emitted_hallucinations")
        assert hasattr(rollup, "n_escaped_hallucinations")
        assert hasattr(rollup, "n_model_generated_unk")
        assert hasattr(rollup, "n_other_failures")

    def test_rollup_has_rate_fields(self):
        rollup = compute_rollup([])
        assert hasattr(rollup, "safe_abstention_rate")
        assert hasattr(rollup, "false_abstention_rate")
        assert hasattr(rollup, "emitted_hallucination_rate_over_generated")
        assert hasattr(rollup, "emitted_hallucination_rate_over_all")
        assert hasattr(rollup, "escaped_hallucination_rate_over_generated")
        assert hasattr(rollup, "escaped_hallucination_rate_over_all")
        assert hasattr(rollup, "constraint_artifact_rate_over_supported_generated")
        assert hasattr(rollup, "accepted_rate_over_all")
        assert hasattr(rollup, "accepted_rate_over_generated")


# ===========================================================================
# §14 — compute_rollup: correct counts
# ===========================================================================

class TestComputeRollupCounts:
    def _make_batch(self) -> list[TranslationEvidence]:
        return [
            _safe_abstention_evidence(item_id="1"),
            _false_abstention_evidence(item_id="2"),
            _base_evidence(item_id="3"),
            _constraint_artifact_evidence(item_id="4"),
            _emitted_hallucination_evidence(item_id="5"),
            _escaped_hallucination_evidence(item_id="6"),
            _model_unk_evidence(item_id="7"),
            _other_failure_evidence(item_id="8"),
        ]

    def test_n_evaluated_correct(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.n_evaluated == 8

    def test_n_generated_correct(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        # backend_called=True for: base, constraint_artifact, emitted, escaped, model_unk, other
        # False for: safe_abstention, false_abstention
        assert rollup.n_generated == 6

    def test_n_supported_correct(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        # source_supported=True for: base, constraint_artifact, emitted, escaped, model_unk, other
        assert rollup.n_supported == 6

    def test_each_outcome_counted_once(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.n_safe_abstentions == 1
        assert rollup.n_false_abstentions == 1
        assert rollup.n_accepted == 1
        assert rollup.n_constraint_artifacts == 1
        assert rollup.n_emitted_hallucinations == 1
        assert rollup.n_escaped_hallucinations == 1
        assert rollup.n_model_generated_unk == 1
        assert rollup.n_other_failures == 1

    def test_counts_sum_to_n_evaluated(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        total = (
            rollup.n_safe_abstentions
            + rollup.n_false_abstentions
            + rollup.n_accepted
            + rollup.n_constraint_artifacts
            + rollup.n_emitted_hallucinations
            + rollup.n_escaped_hallucinations
            + rollup.n_model_generated_unk
            + rollup.n_other_failures
        )
        assert total == rollup.n_evaluated

    def test_empty_batch_all_zero_counts(self):
        rollup = compute_rollup([])
        assert rollup.n_evaluated == 0
        assert rollup.n_generated == 0
        assert rollup.n_supported == 0
        assert rollup.n_accepted == 0

    def test_single_accepted_unit(self):
        rollup = compute_rollup([_base_evidence()])
        assert rollup.n_accepted == 1
        assert rollup.n_emitted_hallucinations == 0
        assert rollup.n_safe_abstentions == 0


# ===========================================================================
# §15 — compute_rollup: rates and explicit denominators
# ===========================================================================

class TestComputeRollupRates:
    def _make_batch(self) -> list[TranslationEvidence]:
        return [
            _safe_abstention_evidence(item_id="1"),
            _false_abstention_evidence(item_id="2"),
            _base_evidence(item_id="3"),
            _constraint_artifact_evidence(item_id="4"),
            _emitted_hallucination_evidence(item_id="5"),
            _escaped_hallucination_evidence(item_id="6"),
            _model_unk_evidence(item_id="7"),
            _other_failure_evidence(item_id="8"),
        ]

    def test_safe_abstention_rate_over_all_evaluated(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.safe_abstention_rate == pytest.approx(1 / 8)

    def test_false_abstention_rate_over_all_evaluated(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.false_abstention_rate == pytest.approx(1 / 8)

    def test_emitted_hallucination_rate_over_generated(self):
        """Denominator: n_generated (backend_called=True units)."""
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        # 1 emitted hallucination / 6 generated
        assert rollup.emitted_hallucination_rate_over_generated == pytest.approx(1 / 6)

    def test_emitted_hallucination_rate_over_all(self):
        """Denominator: n_evaluated."""
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.emitted_hallucination_rate_over_all == pytest.approx(1 / 8)

    def test_escaped_hallucination_rate_over_generated(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.escaped_hallucination_rate_over_generated == pytest.approx(1 / 6)

    def test_escaped_hallucination_rate_over_all(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.escaped_hallucination_rate_over_all == pytest.approx(1 / 8)

    def test_constraint_artifact_rate_over_supported_generated(self):
        """Denominator: units that are both source_supported AND backend_called."""
        # supported AND generated: base, constraint, emitted, escaped, model_unk, other = 6
        # BUT other_failure has source_supported=True and backend_called=True
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.constraint_artifact_rate_over_supported_generated == pytest.approx(1 / 6)

    def test_accepted_rate_over_all(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.accepted_rate_over_all == pytest.approx(1 / 8)

    def test_accepted_rate_over_generated(self):
        batch = self._make_batch()
        rollup = compute_rollup(batch)
        assert rollup.accepted_rate_over_generated == pytest.approx(1 / 6)

    def test_zero_denominator_n_evaluated_returns_zero_not_nan(self):
        """Empty batch: all rates must be 0.0, not NaN."""
        rollup = compute_rollup([])
        assert rollup.safe_abstention_rate == 0.0
        assert rollup.false_abstention_rate == 0.0
        assert rollup.accepted_rate_over_all == 0.0
        assert rollup.accepted_rate_over_generated == 0.0

    def test_zero_denominator_n_generated_returns_zero_not_nan(self):
        """No backend calls: generated-denominator rates are 0.0, not NaN."""
        # Only abstentions — backend never called
        batch = [
            _safe_abstention_evidence(item_id="a"),
            _false_abstention_evidence(item_id="b"),
        ]
        rollup = compute_rollup(batch)
        assert rollup.emitted_hallucination_rate_over_generated == 0.0
        assert rollup.escaped_hallucination_rate_over_generated == 0.0
        assert rollup.constraint_artifact_rate_over_supported_generated == 0.0
        assert rollup.accepted_rate_over_generated == 0.0

    def test_zero_denominator_no_supported_generated_returns_zero(self):
        """No supported+generated units: constraint artifact rate is 0.0."""
        batch = [_safe_abstention_evidence()]
        rollup = compute_rollup(batch)
        assert rollup.constraint_artifact_rate_over_supported_generated == 0.0

    def test_all_accepted_rates_are_one(self):
        batch = [_base_evidence(item_id=str(i)) for i in range(5)]
        rollup = compute_rollup(batch)
        assert rollup.accepted_rate_over_all == pytest.approx(1.0)
        assert rollup.accepted_rate_over_generated == pytest.approx(1.0)

    def test_escaped_hallucination_rate_target_zero(self):
        """Target rate of ESCAPED hallucination must be zero; batch with none gives 0.0."""
        batch = [
            _base_evidence(item_id="a"),
            _emitted_hallucination_evidence(item_id="b"),
        ]
        rollup = compute_rollup(batch)
        assert rollup.escaped_hallucination_rate_over_generated == 0.0
        assert rollup.escaped_hallucination_rate_over_all == 0.0
        assert rollup.n_escaped_hallucinations == 0

    def test_rates_are_fractions_not_percentages(self):
        """Rates are in [0, 1], not [0, 100]."""
        batch = [_base_evidence()]
        rollup = compute_rollup(batch)
        assert 0.0 <= rollup.accepted_rate_over_all <= 1.0
        assert 0.0 <= rollup.accepted_rate_over_generated <= 1.0


# ===========================================================================
# §16 — Sentinel field / wrong-field-use traps
# ===========================================================================

class TestSentinelFieldGuards:
    def test_raw_generation_set_for_backend_units_even_when_rejected(self):
        """raw_generation is the actual backend output, before any audit or serialization."""
        ev = _emitted_hallucination_evidence()
        assert ev.raw_generation is not None
        assert "UNLICENSED" in ev.raw_generation

    def test_surfaced_artifact_distinct_from_raw_generation_for_constraint(self):
        """surfaced_failure_artifact is the failure surface, may equal raw_generation."""
        ev = _constraint_artifact_evidence()
        # Both should be present but reference the right thing
        assert ev.raw_generation is not None
        assert ev.surfaced_failure_artifact is not None

    def test_final_translation_none_is_not_inferred_from_audit_reasons(self):
        """final_translation=None means not accepted, regardless of audit_reasons content."""
        ev = _emitted_hallucination_evidence(audit_reasons=[])
        # Even with empty audit_reasons, if audit_passed=False, should not be ACCEPTED
        assert classify(ev) != Outcome.ACCEPTED_LICENSED

    def test_final_translation_present_with_hallucination_is_escaped_not_accepted(self):
        ev = _escaped_hallucination_evidence()
        assert ev.final_translation is not None
        assert classify(ev) == Outcome.ESCAPED_HALLUCINATION

    def test_model_unk_flag_cannot_be_inferred_from_raw_generation_text(self):
        """model_generated_unk must be an explicit boolean field, not text-parsed."""
        # Evidence has [UNK:...] in raw text but model_generated_unk=False
        # (e.g. it was pre-inserted by UNKDetector)
        ev = _base_evidence(
            raw_generation="Au début [UNK:selah] fut",
            model_generated_unk=False,
            audit_passed=True,
        )
        # Classifier reads the flag, not the text — so it's still ACCEPTED_LICENSED
        assert classify(ev) == Outcome.ACCEPTED_LICENSED

    def test_source_supported_false_and_backend_not_called_explicit_safe_abstention(self):
        """source_supported=False AND backend_called=False is SAFE abstention, not OTHER."""
        ev = _safe_abstention_evidence()
        assert classify(ev) == Outcome.SAFE_SOURCE_ABSTENTION
        assert classify(ev) != Outcome.OTHER_HARD_FAILURE
