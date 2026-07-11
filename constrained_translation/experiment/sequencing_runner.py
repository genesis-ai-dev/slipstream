"""constrained_translation.experiment.sequencing_runner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 5A: Leakage-safe immutable evidence round scheduler.

Provides:
- ``SourceCandidate``: frozen, source-only view of an acquisition candidate.
  No target_texts, no references — safe to pass to policies.
- ``HumanEvidence``: frozen, source + revealed target for one pool item.
  Only ever constructed after a candidate has been selected.
- ``RoundState``: frozen snapshot of one round: evidence pool, selected
  acquisition candidate, optional estimated gain, remaining source-only
  candidates. Never contains future target data.
- ``RoundSchedule``: frozen, ordered sequence of RoundState objects plus
  manifest reference and language tag.
- ``PoolValidationError``: raised on any violation of pool contracts.
- ``validate_pools``: enforce cardinality, uniqueness, nonempty targets,
  and normalized-source-equivalence leakage prohibition.
- ``source_candidates_from_manifest``: adapter — strips target data.
- ``human_evidence_from_manifest``: adapter — builds HumanEvidence for
  a named pool (seed or acquisition).
- ``build_round_schedule``: build schedule from precomputed acq_order.
- ``build_round_schedule_from_selector``: build schedule via per-round
  callback; callback receives only SourceCandidate views.

Design constraints
------------------
- All public data structures are frozen dataclasses or named tuples.
- No persistence, no prediction, no metrics, no plots, no CLI, no GPU.
- Target data is revealed only at the moment a candidate is selected
  (i.e., when adding it to evidence). Remaining candidates are always
  SourceCandidate — no target field ever appears.
- Normalized-source-equivalence leakage prohibition: seed, acquisition,
  and fixed_eval must each have disjoint normalized-source keys. Project
  (remaining) may contain equivalents without violation.
- Pool cardinalities are exact (not "at least") for named pools; project
  uses a minimum threshold (>= expected_project_min).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class PoolValidationError(ValueError):
    """Raised when pool validation constraints are violated."""


# ---------------------------------------------------------------------------
# Immutable data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCandidate:
    """Source-only, leakage-safe view of an acquisition candidate.

    Contains only the fields needed by source-side policies.  Target data
    is deliberately absent to prevent any future-target leakage into
    acquisition decisions.
    """
    corpus_idx: int
    item_id: str
    source_text: str


@dataclass(frozen=True)
class HumanEvidence:
    """Revealed source + target for one pool item.

    Constructed only after a candidate has been selected (i.e., the human
    has translated it).  ``target_text`` must be nonempty — an empty target
    is a data integrity violation.
    """
    corpus_idx: int
    item_id: str
    source_text: str
    target_text: str

    def __post_init__(self) -> None:
        if not self.target_text:
            raise ValueError(
                f"HumanEvidence.target_text must be nonempty; "
                f"corpus_idx={self.corpus_idx!r} has empty target."
            )


@dataclass(frozen=True)
class RoundState:
    """Frozen snapshot of the sequencing scheduler at one round.

    Attributes
    ----------
    round:
        0-based round index.  Round 0 is seed-only.
    evidence:
        Tuple of HumanEvidence items available as few-shot context for
        this round.  Cardinality = len(seed) + round.
    selected:
        The SourceCandidate selected at this round (None for round 0).
        Always a SourceCandidate — never contains target data.
    estimated_gain:
        Optional scalar estimated gain returned by the selector callback
        (None if not provided or if round 0).
    remaining_candidates:
        Tuple of SourceCandidate items not yet revealed at this round.
        Strictly source-only — no target data.
    """
    round: int
    evidence: tuple[HumanEvidence, ...]
    selected: Optional[SourceCandidate]
    estimated_gain: Optional[float]
    remaining_candidates: tuple[SourceCandidate, ...]


@dataclass(frozen=True)
class RoundSchedule:
    """Frozen, ordered sequence of RoundState objects for one (manifest, language) pair.

    Attributes
    ----------
    manifest:
        The SequencingManifest used to build this schedule.
    language:
        Language code for which targets are revealed.
    states:
        Tuple of RoundState; states[0] is round 0 (seed-only),
        states[r] for r >= 1 has r acquisitions revealed.
    """
    manifest: SequencingManifest
    language: str
    states: tuple[RoundState, ...]


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def source_candidates_from_manifest(
    manifest: SequencingManifest,
) -> list[SourceCandidate]:
    """Return SourceCandidates from manifest.acquisition, stripping all target data.

    Parameters
    ----------
    manifest:
        A SequencingManifest produced by build_sequencing_manifest.

    Returns
    -------
    list[SourceCandidate]
        One candidate per acquisition item, in the same order as
        ``manifest.acquisition``.  No target data is carried over.
    """
    return [
        SourceCandidate(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
        )
        for item in manifest.acquisition
    ]


def human_evidence_from_manifest(
    manifest: SequencingManifest,
    language: str,
    pool: str = "seed",
) -> list[HumanEvidence]:
    """Build HumanEvidence list from a named manifest pool.

    Parameters
    ----------
    manifest:
        A SequencingManifest.
    language:
        Language code to extract target text from.
    pool:
        One of "seed" or "acquisition".

    Returns
    -------
    list[HumanEvidence]
        Evidence items in manifest pool order, with source + target.

    Raises
    ------
    ValueError
        If ``pool`` is not recognized.
    KeyError
        If ``language`` is not in item.target_texts.
    """
    pool_map: dict[str, list[SequencingPoolItem]] = {
        "seed": manifest.seed,
        "acquisition": manifest.acquisition,
    }
    if pool not in pool_map:
        raise ValueError(
            f"Unknown pool {pool!r}; expected one of {sorted(pool_map)}"
        )
    items = pool_map[pool]
    return [
        HumanEvidence(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
            target_text=item.target_texts[language],
        )
        for item in items
    ]


# ---------------------------------------------------------------------------
# Internal helper: normalized source key
# ---------------------------------------------------------------------------

def _norm_key(text: str) -> str:
    """Return a normalized source equivalence key for the given text."""
    return " ".join(normalize_source_units(text))


# ---------------------------------------------------------------------------
# Pool validation
# ---------------------------------------------------------------------------

def validate_pools(
    manifest: SequencingManifest,
    language: str,
    *,
    expected_seed: int,
    expected_acq: int,
    expected_eval: int,
    expected_project_min: int,
) -> None:
    """Validate all pool contracts for the given manifest and language.

    Enforces:
    1. Exact cardinalities for seed, acquisition, fixed_eval.
    2. Minimum cardinality for remaining (project).
    3. Unique corpus_idx within seed and within acquisition.
    4. No corpus_idx overlap between seed and acquisition.
    5. Nonempty human target for every seed and acquisition item.
    6. Language key present in every seed and acquisition item.
    7. Normalized-source-equivalence disjointness among seed, acquisition,
       and fixed_eval (no normalized key crosses among these three named pools).
    8. Remaining/project may contain equivalents without violation.

    Parameters
    ----------
    manifest:
        Manifest to validate.
    language:
        Language code to check for target presence.
    expected_seed:
        Exact required cardinality of manifest.seed.
    expected_acq:
        Exact required cardinality of manifest.acquisition.
    expected_eval:
        Exact required cardinality of manifest.fixed_eval.
    expected_project_min:
        Minimum required cardinality of manifest.remaining.

    Raises
    ------
    PoolValidationError
        On any contract violation.
    """
    # 1. Exact cardinalities
    if len(manifest.seed) != expected_seed:
        raise PoolValidationError(
            f"Seed pool has {len(manifest.seed)} items; expected exactly {expected_seed}."
        )
    if len(manifest.acquisition) != expected_acq:
        raise PoolValidationError(
            f"Acquisition pool has {len(manifest.acquisition)} items; expected exactly {expected_acq}."
        )
    if len(manifest.fixed_eval) != expected_eval:
        raise PoolValidationError(
            f"Fixed-eval pool has {len(manifest.fixed_eval)} items; expected exactly {expected_eval}."
        )
    # 2. Minimum project cardinality
    if len(manifest.remaining) < expected_project_min:
        raise PoolValidationError(
            f"Remaining/project pool has {len(manifest.remaining)} items; "
            f"need at least {expected_project_min}."
        )

    # 3. Unique corpus_idx within seed
    seed_idxs = [item.corpus_idx for item in manifest.seed]
    if len(seed_idxs) != len(set(seed_idxs)):
        raise PoolValidationError(
            f"Seed pool has duplicate corpus_idx values: {seed_idxs}"
        )

    # 4. Unique corpus_idx within acquisition
    acq_idxs = [item.corpus_idx for item in manifest.acquisition]
    if len(acq_idxs) != len(set(acq_idxs)):
        raise PoolValidationError(
            f"Acquisition pool has duplicate corpus_idx values: {acq_idxs}"
        )

    # 5. No seed/acquisition corpus_idx overlap
    seed_idx_set = set(seed_idxs)
    acq_idx_set = set(acq_idxs)
    overlap = seed_idx_set & acq_idx_set
    if overlap:
        raise PoolValidationError(
            f"Seed and acquisition share corpus_idx values: {sorted(overlap)}"
        )

    # 6+7. Language key and nonempty target in seed and acquisition
    for pool_name, items in [("seed", manifest.seed), ("acquisition", manifest.acquisition)]:
        for item in items:
            if language not in item.target_texts:
                raise PoolValidationError(
                    f"{pool_name} item corpus_idx={item.corpus_idx} is missing "
                    f"language {language!r} in target_texts."
                )
            if not item.target_texts[language]:
                raise PoolValidationError(
                    f"{pool_name} item corpus_idx={item.corpus_idx} has empty "
                    f"target_text for language {language!r}."
                )

    # 8. Normalized-source-equivalence leakage prohibition among seed/acq/eval.
    # Each of the three named pools must have disjoint normalized keys;
    # no key may cross among them.
    seed_keys: dict[str, int] = {}
    for item in manifest.seed:
        k = _norm_key(item.source_text)
        seed_keys[k] = item.corpus_idx

    acq_keys: dict[str, int] = {}
    for item in manifest.acquisition:
        k = _norm_key(item.source_text)
        acq_keys[k] = item.corpus_idx

    eval_keys: dict[str, int] = {}
    for item in manifest.fixed_eval:
        k = _norm_key(item.source_text)
        eval_keys[k] = item.corpus_idx

    # seed ∩ acq
    shared_seed_acq = set(seed_keys) & set(acq_keys)
    if shared_seed_acq:
        raise PoolValidationError(
            f"Normalized-source-equivalent leakage: seed and acquisition share "
            f"{len(shared_seed_acq)} norm key(s): {sorted(shared_seed_acq)[:3]}..."
        )

    # seed ∩ eval
    shared_seed_eval = set(seed_keys) & set(eval_keys)
    if shared_seed_eval:
        raise PoolValidationError(
            f"Normalized-source-equivalent leakage: seed and fixed_eval share "
            f"{len(shared_seed_eval)} norm key(s): {sorted(shared_seed_eval)[:3]}..."
        )

    # acq ∩ eval
    shared_acq_eval = set(acq_keys) & set(eval_keys)
    if shared_acq_eval:
        raise PoolValidationError(
            f"Normalized-source-equivalent leakage: acquisition and fixed_eval share "
            f"{len(shared_acq_eval)} norm key(s): {sorted(shared_acq_eval)[:3]}..."
        )


# ---------------------------------------------------------------------------
# Internal: build acquisition lookup map
# ---------------------------------------------------------------------------

def _build_acq_lookup(
    manifest: SequencingManifest,
    language: str,
) -> dict[int, SequencingPoolItem]:
    """Return corpus_idx → SequencingPoolItem for all acquisition items."""
    return {item.corpus_idx: item for item in manifest.acquisition}


# ---------------------------------------------------------------------------
# Build schedule from precomputed order
# ---------------------------------------------------------------------------

def build_round_schedule(
    manifest: SequencingManifest,
    language: str,
    acq_order: list[int],
    *,
    expected_seed: int,
    expected_acq: int,
    expected_eval: int,
    expected_project_min: int,
) -> RoundSchedule:
    """Build an immutable RoundSchedule from a precomputed acquisition order.

    Parameters
    ----------
    manifest:
        Sequencing manifest defining the pools.
    language:
        Language code for target revelation.
    acq_order:
        Ordered list of corpus_idx values specifying the acquisition
        reveal sequence.  Must contain each acquisition corpus_idx
        exactly once.
    expected_seed, expected_acq, expected_eval, expected_project_min:
        Passed verbatim to ``validate_pools``.

    Returns
    -------
    RoundSchedule
        Frozen schedule with len(acq_order) + 1 states.

    Raises
    ------
    PoolValidationError
        If pool validation fails.
    ValueError
        If acq_order contains duplicates, invalid corpus_idx values, or
        does not cover all acquisition items.
    """
    # Validate pools first
    validate_pools(
        manifest, language,
        expected_seed=expected_seed,
        expected_acq=expected_acq,
        expected_eval=expected_eval,
        expected_project_min=expected_project_min,
    )

    acq_lookup = _build_acq_lookup(manifest, language)
    acq_corpus_idx_set = set(acq_lookup)

    # Validate acq_order
    if len(acq_order) != len(set(acq_order)):
        raise ValueError(
            f"acq_order contains duplicate corpus_idx values: {acq_order}"
        )
    for idx in acq_order:
        if idx not in acq_corpus_idx_set:
            raise ValueError(
                f"acq_order contains corpus_idx={idx} which is not in "
                f"manifest.acquisition (valid: {sorted(acq_corpus_idx_set)})"
            )
    if set(acq_order) != acq_corpus_idx_set:
        missing = acq_corpus_idx_set - set(acq_order)
        raise ValueError(
            f"acq_order does not cover all acquisition items; missing: {sorted(missing)}"
        )

    # Build seed evidence
    seed_evidence: tuple[HumanEvidence, ...] = tuple(
        HumanEvidence(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
            target_text=item.target_texts[language],
        )
        for item in manifest.seed
    )

    # Build initial remaining candidates (all acquisition, as SourceCandidate)
    all_candidates: dict[int, SourceCandidate] = {
        item.corpus_idx: SourceCandidate(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
        )
        for item in manifest.acquisition
    }

    states: list[RoundState] = []
    current_evidence: tuple[HumanEvidence, ...] = seed_evidence
    remaining_set: dict[int, SourceCandidate] = dict(all_candidates)

    # Round 0: seed-only
    round0 = RoundState(
        round=0,
        evidence=current_evidence,
        selected=None,
        estimated_gain=None,
        remaining_candidates=tuple(
            remaining_set[idx] for idx in sorted(remaining_set)
        ),
    )
    states.append(round0)

    # Rounds 1..N: reveal one acquisition each
    for r, corpus_idx in enumerate(acq_order, start=1):
        item = acq_lookup[corpus_idx]
        selected_sc = all_candidates[corpus_idx]
        revealed = HumanEvidence(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
            target_text=item.target_texts[language],
        )
        current_evidence = current_evidence + (revealed,)
        del remaining_set[corpus_idx]

        state = RoundState(
            round=r,
            evidence=current_evidence,
            selected=selected_sc,
            estimated_gain=None,
            remaining_candidates=tuple(
                remaining_set[idx] for idx in sorted(remaining_set)
            ),
        )
        states.append(state)

    return RoundSchedule(
        manifest=manifest,
        language=language,
        states=tuple(states),
    )


# ---------------------------------------------------------------------------
# Build schedule from selector callback
# ---------------------------------------------------------------------------

# Type alias for selector callback
SelectorCallback = Callable[
    [list[SourceCandidate], list[HumanEvidence]],
    tuple[int, Optional[float]],
]


def build_round_schedule_from_selector(
    manifest: SequencingManifest,
    language: str,
    selector: SelectorCallback,
    *,
    expected_seed: int,
    expected_acq: int,
    expected_eval: int,
    expected_project_min: int,
) -> RoundSchedule:
    """Build an immutable RoundSchedule via a per-round selector callback.

    The selector is invoked once per acquisition round with:
    - ``remaining``: list of SourceCandidate (source-only; no target data).
    - ``evidence``: list of HumanEvidence (current revealed evidence pool,
      i.e. the evidence from the *previous* round before the new reveal).

    The selector returns ``(corpus_idx, estimated_gain)`` where:
    - ``corpus_idx`` must be the corpus_idx of one of the remaining candidates.
    - ``estimated_gain`` may be None or a float.

    Parameters
    ----------
    manifest, language, expected_*:
        As in ``build_round_schedule``.
    selector:
        Callback ``(remaining: list[SourceCandidate], evidence: list[HumanEvidence])
        -> (corpus_idx: int, estimated_gain: Optional[float])``.

    Returns
    -------
    RoundSchedule

    Raises
    ------
    PoolValidationError
        If pool validation fails.
    ValueError
        If the selector returns a corpus_idx not in remaining, or
        selects a duplicate.
    """
    # Validate pools
    validate_pools(
        manifest, language,
        expected_seed=expected_seed,
        expected_acq=expected_acq,
        expected_eval=expected_eval,
        expected_project_min=expected_project_min,
    )

    acq_lookup = _build_acq_lookup(manifest, language)

    # Build seed evidence
    seed_evidence: tuple[HumanEvidence, ...] = tuple(
        HumanEvidence(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
            target_text=item.target_texts[language],
        )
        for item in manifest.seed
    )

    # Build all candidates as SourceCandidate (source-only)
    all_candidates: dict[int, SourceCandidate] = {
        item.corpus_idx: SourceCandidate(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
        )
        for item in manifest.acquisition
    }

    states: list[RoundState] = []
    current_evidence: tuple[HumanEvidence, ...] = seed_evidence
    remaining_set: dict[int, SourceCandidate] = dict(all_candidates)
    revealed_idxs: set[int] = set()

    # Round 0
    round0 = RoundState(
        round=0,
        evidence=current_evidence,
        selected=None,
        estimated_gain=None,
        remaining_candidates=tuple(
            remaining_set[idx] for idx in sorted(remaining_set)
        ),
    )
    states.append(round0)

    # Rounds 1..N
    for r in range(1, len(manifest.acquisition) + 1):
        remaining_list = [remaining_set[idx] for idx in sorted(remaining_set)]
        evidence_list = list(current_evidence)

        # Call selector with source-only view
        result = selector(remaining_list, evidence_list)
        chosen_idx, estimated_gain = result[0], result[1] if len(result) > 1 else None

        # Validate selector choice
        if chosen_idx not in remaining_set:
            raise ValueError(
                f"Selector returned corpus_idx={chosen_idx} which is not in "
                f"remaining candidates {sorted(remaining_set.keys())}"
            )
        if chosen_idx in revealed_idxs:
            raise ValueError(
                f"Selector returned corpus_idx={chosen_idx} which was already revealed."
            )

        selected_sc = all_candidates[chosen_idx]
        item = acq_lookup[chosen_idx]
        revealed = HumanEvidence(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
            target_text=item.target_texts[language],
        )
        current_evidence = current_evidence + (revealed,)
        del remaining_set[chosen_idx]
        revealed_idxs.add(chosen_idx)

        state = RoundState(
            round=r,
            evidence=current_evidence,
            selected=selected_sc,
            estimated_gain=estimated_gain,
            remaining_candidates=tuple(
                remaining_set[idx] for idx in sorted(remaining_set)
            ),
        )
        states.append(state)

    return RoundSchedule(
        manifest=manifest,
        language=language,
        states=tuple(states),
    )
