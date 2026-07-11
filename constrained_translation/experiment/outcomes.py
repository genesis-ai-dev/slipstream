"""constrained_translation.experiment.outcomes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Explicit safety and constraint-cost outcome taxonomy (Task 4).

Taxonomy
--------
Eight mutually exclusive primary outcomes:

1. SAFE_SOURCE_ABSTENTION    Deterministic pre-generation: source evidence
                              is unsupported, backend not called. Safe behaviour,
                              not a defect.
2. FALSE_SOURCE_ABSTENTION   Detector reports unsupported although a canonical
                              unit exists in translated source evidence. Engineering
                              / normalisation defect.
3. ACCEPTED_LICENSED         Source supported, backend called, all audits passed,
                              no model-generated UNK, and final_translation is not None.
4. CONSTRAINT_ARTIFACT       Source supported; visible lexical surface is licensed;
                              request/output rejected solely by grammar, tokeniser,
                              byte-decode, terminal-control, layout, or normalisation
                              machinery. Must NOT swallow an actual unlicensed word.
5. EMITTED_HALLUCINATION     Model emitted an unlicensed token ID or unattested
                              visible surface / recombination. Rejected failure.
6. ESCAPED_HALLUCINATION     Any emitted hallucination that would be accepted /
                              serialised as the final translation. Invariant
                              violation: classified explicitly and serialisation
                              guard raises EscapedHallucinationError. Target rate
                              must remain zero.
7. MODEL_GENERATED_UNK       Model spontaneously emitted a [UNK:…] marker.
                              Separate rejected violation; deterministic UNK
                              insertion may only occur before generation.
8. OTHER_HARD_FAILURE        Backend / grammar / request failure that fits none
                              of the above. Must not relabel any classified case.

Classification decision order (priority descending)
----------------------------------------------------
1.  false_abstention_detected → FALSE_SOURCE_ABSTENTION
2.  not source_supported AND not backend_called → SAFE_SOURCE_ABSTENTION
3.  backend_called AND raw_generation is None → OTHER_HARD_FAILURE
3b. backend_called AND (model_generated_unk OR raw_generation contains [UNK:<src>]) → MODEL_GENERATED_UNK
4.  backend_called AND (not exact_id_licensed OR not visible_surface_licensed):
        if final_translation is not None → ESCAPED_HALLUCINATION
        else                             → EMITTED_HALLUCINATION
5.  backend_called AND not audit_passed AND (exact_id_licensed AND visible_surface_licensed)
        → CONSTRAINT_ARTIFACT
6.  backend_called AND audit_passed AND source_supported AND not model_generated_unk
        AND final_translation is not None → ACCEPTED_LICENSED
7.  everything else → OTHER_HARD_FAILURE

Design notes
------------
- ``classify`` is a pure function over an explicit ``TranslationEvidence``
  dataclass. No brittle message-substring matching is used to determine the
  primary outcome; the classifier reads typed sentinel fields.
- ``audit_reasons`` is available for downstream diagnostics but is NOT parsed
  to derive the primary outcome.
- ``accepted`` is NEVER inferred from a non-empty ``final_translation`` string.
- Rollup denominators are explicit integers; zero denominators return 0.0
  (never NaN).
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Reserved UNK marker detector
# ---------------------------------------------------------------------------

# The reserved marker syntax is [UNK:<source>] where <source> is non-empty.
# This is case-sensitive: only uppercase UNK is the reserved form.
# [unk:x] or [UNK] (no colon/source) are not reserved markers.
_RESERVED_UNK_MARKER_RE = re.compile(r'\[UNK:[^\]]+\]')


def _raw_contains_reserved_unk_marker(text: str) -> bool:
    """Return True iff *text* contains the reserved [UNK:<source>] marker syntax.

    Narrow, explicit detector — not broad substring matching.  Only matches
    the exact form ``[UNK:<non-empty-source>]`` (case-sensitive uppercase UNK).
    """
    return bool(_RESERVED_UNK_MARKER_RE.search(text))


# ---------------------------------------------------------------------------
# Outcome enum
# ---------------------------------------------------------------------------

class Outcome(enum.Enum):
    """Mutually exclusive primary translation outcome."""

    SAFE_SOURCE_ABSTENTION = "SAFE_SOURCE_ABSTENTION"
    FALSE_SOURCE_ABSTENTION = "FALSE_SOURCE_ABSTENTION"
    ACCEPTED_LICENSED = "ACCEPTED_LICENSED"
    CONSTRAINT_ARTIFACT = "CONSTRAINT_ARTIFACT"
    EMITTED_HALLUCINATION = "EMITTED_HALLUCINATION"
    ESCAPED_HALLUCINATION = "ESCAPED_HALLUCINATION"
    MODEL_GENERATED_UNK = "MODEL_GENERATED_UNK"
    OTHER_HARD_FAILURE = "OTHER_HARD_FAILURE"


# ---------------------------------------------------------------------------
# TranslationEvidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranslationEvidence:
    """All evidence required to classify one translation unit's outcome.

    Sentinel fields
    ---------------
    raw_generation
        The backend's verbatim output text (before audit or serialisation).
        None when backend was not called.

    surfaced_failure_artifact
        The specific text surface that caused the rejection (may equal
        raw_generation). Preserved for downstream diagnostics even when
        final_translation is None (rejected). Must not be confused with
        final_translation.

    final_translation
        The accepted, serialised translation text. None for any rejected
        outcome including CONSTRAINT_ARTIFACT, EMITTED_HALLUCINATION,
        MODEL_GENERATED_UNK, and all abstentions. NOT used to infer
        acceptance — only audit_passed determines acceptance.

    Audit booleans / reasons
    ------------------------
    exact_id_licensed
        True iff every generated token ID is present in the token licence
        built from the example targets.

    visible_surface_licensed
        True iff every surface word in the full-sequence-decoded output is
        present in attested_vocab (surface-composition check).

    full_decode_match
        True iff the full-sequence decode of the generated token IDs matches
        the raw_generation text exactly (byte-for-byte).

    control_token_valid
        True iff terminal-control tokens (e.g. <|im_end|>) appear only at the
        final position (or not at all). False if a control token appears at a
        non-final position.

    grammar_accepted
        True iff the generation satisfied the XGrammar constraint without any
        grammar-level rejection.

    audit_passed
        True iff the TokenAuditResult.passed is True — all sub-checks passed.

    audit_reasons
        List of human-readable violation strings from TokenAuditResult.violations.
        For diagnostic use only; not parsed by classify().

    model_generated_unk
        True iff the model spontaneously emitted a [UNK:…] marker. The
        classifier reads this typed boolean field AND also checks raw_generation
        text for the reserved [UNK:<source>] marker syntax (case-sensitive) as a
        safety override — if raw_generation contains the marker and backend was
        called, MODEL_GENERATED_UNK is returned regardless of this flag's value.
        Deterministic pre-generation UNK insertion only occurs before backend call
        so the marker cannot legitimately appear in raw_generation post-generation.

    Source coverage
    ---------------
    source_supported
        True iff the source-side coverage check found at least one unit in
        the attested vocabulary (pre-generation gate).

    false_abstention_detected
        True iff the source-coverage detector returned unsupported although
        at least one canonical unit from the source IS present in the
        translated source evidence. Indicates a normalisation/index defect.

    Backend
    -------
    backend_called
        True iff a generation backend was actually invoked for this unit.
    """

    item_id: str
    source_text: str

    # Source coverage
    source_supported: bool
    false_abstention_detected: bool

    # Backend
    backend_called: bool

    # Raw generation sentinel
    raw_generation: Optional[str]

    # Audit booleans
    exact_id_licensed: bool
    visible_surface_licensed: bool
    full_decode_match: bool
    control_token_valid: bool
    grammar_accepted: bool
    audit_passed: bool
    audit_reasons: List[str]
    model_generated_unk: bool

    # Failure artifact sentinel (raw rejected text)
    surfaced_failure_artifact: Optional[str]

    # Final accepted translation sentinel (None when rejected)
    final_translation: Optional[str]


# ---------------------------------------------------------------------------
# EscapedHallucinationError
# ---------------------------------------------------------------------------

class EscapedHallucinationError(RuntimeError):
    """Raised when an emitted hallucination would be serialised as the final
    translation.  This is an invariant violation; the target rate must be zero.

    The error message includes the item_id for easy tracing.
    """


class MissingForensicArtifactError(RuntimeError):
    """Raised by ``guard_forensic_artifact_present`` when a rejected generated
    outcome (EMITTED_HALLUCINATION, CONSTRAINT_ARTIFACT, MODEL_GENERATED_UNK,
    ESCAPED_HALLUCINATION) is missing its required ``surfaced_failure_artifact``.

    Rejected outcomes require the forensic artifact for downstream diagnostics
    and audit trails. Absence is a loud invariant violation; callers must supply
    the rejected text before the outcome is finalised.

    The error message includes the item_id and outcome name for easy tracing.
    """


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def classify(evidence: TranslationEvidence) -> Outcome:
    """Classify *evidence* into exactly one mutually exclusive ``Outcome``.

    Classification priority (descending):

    1. ``false_abstention_detected`` → FALSE_SOURCE_ABSTENTION
    2. ``not source_supported and not backend_called`` → SAFE_SOURCE_ABSTENTION
    3. ``backend_called and raw_generation is None`` → OTHER_HARD_FAILURE
    3b.``backend_called and (model_generated_unk or raw_generation contains [UNK:<src>])``
           → MODEL_GENERATED_UNK
       Raw-marker check is a safety override: deterministic pre-generation UNK
       insertion occurs before the backend call, so the reserved marker cannot
       legitimately appear in raw_generation post-generation.
    4. ``backend_called and (not exact_id_licensed or not visible_surface_licensed)``:
           → ESCAPED_HALLUCINATION if ``final_translation is not None``
           → EMITTED_HALLUCINATION otherwise
    5. ``backend_called and not audit_passed and exact_id_licensed and visible_surface_licensed``
           → CONSTRAINT_ARTIFACT
    6. ``backend_called and audit_passed and source_supported and not model_generated_unk
          and final_translation is not None``
           → ACCEPTED_LICENSED
    7. everything else → OTHER_HARD_FAILURE

    Parameters
    ----------
    evidence:
        Populated ``TranslationEvidence`` for one translation unit.

    Returns
    -------
    Outcome
        Exactly one of the eight primary outcomes.
    """
    ev = evidence

    # Priority 1: engineering/normalisation defect supersedes all else
    if ev.false_abstention_detected:
        return Outcome.FALSE_SOURCE_ABSTENTION

    # Priority 2: deterministic pre-generation unsupported source, no backend call
    if not ev.source_supported and not ev.backend_called:
        return Outcome.SAFE_SOURCE_ABSTENTION

    # All remaining priorities require backend_called=True
    if ev.backend_called:

        # Priority 3: backend was called but produced no raw output (crash/timeout)
        # This is a hard failure; no model output means no hallucination to classify.
        if ev.raw_generation is None:
            return Outcome.OTHER_HARD_FAILURE

        # Priority 3b: model spontaneously emitted [UNK:…]
        # Safety override: if raw_generation contains the reserved marker syntax,
        # force MODEL_GENERATED_UNK even when model_generated_unk flag is False.
        _raw_has_marker = _raw_contains_reserved_unk_marker(ev.raw_generation)
        if ev.model_generated_unk or _raw_has_marker:
            return Outcome.MODEL_GENERATED_UNK

        # Priority 4: unlicensed token ID or unattested visible surface
        if not ev.exact_id_licensed or not ev.visible_surface_licensed:
            if ev.final_translation is not None:
                return Outcome.ESCAPED_HALLUCINATION
            return Outcome.EMITTED_HALLUCINATION

        # Priority 5: surface is licensed but machinery (grammar/tokeniser/control) rejected
        if not ev.audit_passed and ev.exact_id_licensed and ev.visible_surface_licensed:
            return Outcome.CONSTRAINT_ARTIFACT

        # Priority 6: clean acceptance — requires final_translation is not None
        if (ev.audit_passed and ev.source_supported and not ev.model_generated_unk
                and ev.final_translation is not None):
            return Outcome.ACCEPTED_LICENSED

    # Priority 7: none of the above fit
    return Outcome.OTHER_HARD_FAILURE


# ---------------------------------------------------------------------------
# guard_no_escaped_hallucination
# ---------------------------------------------------------------------------

def guard_no_escaped_hallucination(
    outcome: Outcome,
    evidence: TranslationEvidence,
) -> None:
    """Raise ``EscapedHallucinationError`` if *outcome* is ESCAPED_HALLUCINATION.

    Call this before serialising / returning any translation to catch invariant
    violations loudly.  For all other outcomes this function is a no-op.

    Parameters
    ----------
    outcome:
        The result of ``classify(evidence)``.
    evidence:
        The same ``TranslationEvidence`` used to derive *outcome*.  Used to
        construct an informative error message.

    Raises
    ------
    EscapedHallucinationError
        When *outcome* is ``Outcome.ESCAPED_HALLUCINATION``.
    """
    if outcome is Outcome.ESCAPED_HALLUCINATION:
        raise EscapedHallucinationError(
            f"ESCAPED_HALLUCINATION detected for item {evidence.item_id!r}: "
            f"an unlicensed generation was serialised as the final translation. "
            f"raw_generation={evidence.raw_generation!r}  "
            f"final_translation={evidence.final_translation!r}  "
            f"audit_reasons={evidence.audit_reasons!r}"
        )


# ---------------------------------------------------------------------------
# guard_forensic_artifact_present
# ---------------------------------------------------------------------------

# Outcomes that require surfaced_failure_artifact to be set (not None).
_FORENSIC_ARTIFACT_REQUIRED: frozenset[Outcome] = frozenset({
    Outcome.EMITTED_HALLUCINATION,
    Outcome.CONSTRAINT_ARTIFACT,
    Outcome.MODEL_GENERATED_UNK,
    Outcome.ESCAPED_HALLUCINATION,
})


def guard_forensic_artifact_present(
    outcome: Outcome,
    evidence: TranslationEvidence,
) -> None:
    """Raise ``MissingForensicArtifactError`` if *outcome* requires a forensic
    artifact but ``evidence.surfaced_failure_artifact`` is ``None``.

    Rejected generated outcomes (EMITTED_HALLUCINATION, CONSTRAINT_ARTIFACT,
    MODEL_GENERATED_UNK, ESCAPED_HALLUCINATION) must retain the rejected text
    in ``surfaced_failure_artifact`` for downstream diagnostics and audit trails.
    For ESCAPED_HALLUCINATION the guard additionally verifies that the raw
    evidence is preserved (raw_generation is available on the evidence object).

    For all other outcomes this function is a no-op.

    Parameters
    ----------
    outcome:
        The result of ``classify(evidence)``.
    evidence:
        The same ``TranslationEvidence`` used to derive *outcome*.

    Raises
    ------
    MissingForensicArtifactError
        When *outcome* is one of the four rejection outcomes and
        ``evidence.surfaced_failure_artifact`` is ``None``.
    """
    if outcome in _FORENSIC_ARTIFACT_REQUIRED:
        if evidence.surfaced_failure_artifact is None:
            raise MissingForensicArtifactError(
                f"Missing surfaced_failure_artifact for {outcome.value} "
                f"on item {evidence.item_id!r}. "
                f"Rejected outcomes must retain the forensic text artifact for diagnostics. "
                f"raw_generation={evidence.raw_generation!r}"
            )


# ---------------------------------------------------------------------------
# OutcomeRollup dataclass
# ---------------------------------------------------------------------------

@dataclass
class OutcomeRollup:
    """Aggregate counts and rates across a batch of ``TranslationEvidence`` records.

    Denominator definitions
    -----------------------
    n_evaluated
        All units evaluated (== len(batch)).

    n_generated
        Units for which the backend was actually called
        (``backend_called=True``). This denominator includes ALL backend-called
        units regardless of their final outcome, including those that end in
        OTHER_HARD_FAILURE. It therefore represents the total generation attempts
        made by the backend, not just successful or clean completions.

    n_supported
        Units where ``source_supported=True``.

    n_supported_generated
        Units where both ``source_supported=True`` and ``backend_called=True``.
        Used as the denominator for constraint_artifact_rate.

    Rate denominators (explicit, never NaN)
    ----------------------------------------
    safe_abstention_rate            / n_evaluated
    false_abstention_rate           / n_evaluated
    emitted_hallucination_rate_over_generated   / n_generated
    emitted_hallucination_rate_over_all         / n_evaluated
    escaped_hallucination_rate_over_generated   / n_generated
    escaped_hallucination_rate_over_all         / n_evaluated
    constraint_artifact_rate_over_supported_generated / n_supported_generated
    accepted_rate_over_all          / n_evaluated
    accepted_rate_over_generated    / n_generated

    All rates are in [0.0, 1.0]. Zero denominators → 0.0 (not NaN).
    """

    # Denominators
    n_evaluated: int
    n_generated: int
    n_supported: int
    n_supported_generated: int

    # Counts
    n_safe_abstentions: int
    n_false_abstentions: int
    n_accepted: int
    n_constraint_artifacts: int
    n_emitted_hallucinations: int
    n_escaped_hallucinations: int
    n_model_generated_unk: int
    n_other_failures: int

    # Rates
    safe_abstention_rate: float
    false_abstention_rate: float
    emitted_hallucination_rate_over_generated: float
    emitted_hallucination_rate_over_all: float
    escaped_hallucination_rate_over_generated: float
    escaped_hallucination_rate_over_all: float
    constraint_artifact_rate_over_supported_generated: float
    accepted_rate_over_all: float
    accepted_rate_over_generated: float


# ---------------------------------------------------------------------------
# compute_rollup
# ---------------------------------------------------------------------------

def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_rollup(batch: Sequence[TranslationEvidence]) -> OutcomeRollup:
    """Compute aggregate outcome counts and rates for *batch*.

    Parameters
    ----------
    batch:
        Sequence of ``TranslationEvidence`` objects (may be empty).

    Returns
    -------
    OutcomeRollup
        All counts and rates with explicit denominators. Zero denominators
        produce 0.0 rates (never NaN).
    """
    n_evaluated = len(batch)

    # Classify each evidence item
    outcomes: list[Outcome] = [classify(ev) for ev in batch]

    # Denominators
    n_generated = sum(1 for ev in batch if ev.backend_called)
    n_supported = sum(1 for ev in batch if ev.source_supported)
    n_supported_generated = sum(
        1 for ev in batch if ev.source_supported and ev.backend_called
    )

    # Counts per outcome
    n_safe_abstentions = outcomes.count(Outcome.SAFE_SOURCE_ABSTENTION)
    n_false_abstentions = outcomes.count(Outcome.FALSE_SOURCE_ABSTENTION)
    n_accepted = outcomes.count(Outcome.ACCEPTED_LICENSED)
    n_constraint_artifacts = outcomes.count(Outcome.CONSTRAINT_ARTIFACT)
    n_emitted_hallucinations = outcomes.count(Outcome.EMITTED_HALLUCINATION)
    n_escaped_hallucinations = outcomes.count(Outcome.ESCAPED_HALLUCINATION)
    n_model_generated_unk = outcomes.count(Outcome.MODEL_GENERATED_UNK)
    n_other_failures = outcomes.count(Outcome.OTHER_HARD_FAILURE)

    # Rates (zero denominators → 0.0)
    safe_abstention_rate = _safe_rate(n_safe_abstentions, n_evaluated)
    false_abstention_rate = _safe_rate(n_false_abstentions, n_evaluated)
    emitted_hallucination_rate_over_generated = _safe_rate(n_emitted_hallucinations, n_generated)
    emitted_hallucination_rate_over_all = _safe_rate(n_emitted_hallucinations, n_evaluated)
    escaped_hallucination_rate_over_generated = _safe_rate(n_escaped_hallucinations, n_generated)
    escaped_hallucination_rate_over_all = _safe_rate(n_escaped_hallucinations, n_evaluated)
    constraint_artifact_rate_over_supported_generated = _safe_rate(
        n_constraint_artifacts, n_supported_generated
    )
    accepted_rate_over_all = _safe_rate(n_accepted, n_evaluated)
    accepted_rate_over_generated = _safe_rate(n_accepted, n_generated)

    return OutcomeRollup(
        n_evaluated=n_evaluated,
        n_generated=n_generated,
        n_supported=n_supported,
        n_supported_generated=n_supported_generated,
        n_safe_abstentions=n_safe_abstentions,
        n_false_abstentions=n_false_abstentions,
        n_accepted=n_accepted,
        n_constraint_artifacts=n_constraint_artifacts,
        n_emitted_hallucinations=n_emitted_hallucinations,
        n_escaped_hallucinations=n_escaped_hallucinations,
        n_model_generated_unk=n_model_generated_unk,
        n_other_failures=n_other_failures,
        safe_abstention_rate=safe_abstention_rate,
        false_abstention_rate=false_abstention_rate,
        emitted_hallucination_rate_over_generated=emitted_hallucination_rate_over_generated,
        emitted_hallucination_rate_over_all=emitted_hallucination_rate_over_all,
        escaped_hallucination_rate_over_generated=escaped_hallucination_rate_over_generated,
        escaped_hallucination_rate_over_all=escaped_hallucination_rate_over_all,
        constraint_artifact_rate_over_supported_generated=constraint_artifact_rate_over_supported_generated,
        accepted_rate_over_all=accepted_rate_over_all,
        accepted_rate_over_generated=accepted_rate_over_generated,
    )
