"""tests/constrained_translation/experiment/test_sequencing_runner.py

Strict TDD tests for Task 5A: leakage-safe immutable evidence round scheduler.

Written BEFORE the implementation (RED phase).

Required behaviour (per docs/plans/2026-07-10-slipstream-sequencing.md §Task 5
and task scope):

1. Minimal immutable source/human evidence models:
   - SourceCandidate: corpus_idx, item_id, source_text (no target data).
   - HumanEvidence: corpus_idx, item_id, source_text, target_text (per-language).
   - Adapter functions to build these from manifest records without target leakage
     into candidate views.

2. Pool validation:
   - Production defaults: seed=10, acquisition=40, project=200, fixed_eval=60
   - Configurable expected cardinalities for tiny tests.
   - Stable unique item_ids and corpus_indices.
   - Every seed/acquisition item has a nonempty human target (target_text).
   - No seed/acquisition overlap (by corpus_idx).
   - Project pool from remaining (first N by corpus_idx).
   - Fixed eval follows manifest fixed_eval.
   - Normalized-source-equivalent leakage prohibition: seed, acquisition, and
     fixed_eval must each have distinct normalized-source keys; no key may cross
     among these three named pools (verified via validation function).
   - Project (remaining) may contain real duplicates/equivalents as per brief.

3. RoundSchedule / immutable RoundState:
   - Round 0: seed evidence only; 0 acquisitions revealed.
   - Round r (1..N): exactly one new acquisition target revealed vs. round r-1.
   - Evidence cardinality at round r = len(seed) + r.
   - Selected acquisition item_id and corpus_idx included in state metadata.
   - Optional estimated_gain field in state metadata.
   - Remaining source-only SourceCandidate list (no target refs in candidates).
   - No future target data in policy candidate view.

4. Two acquisition-order inputs:
   - Precomputed order: list[int] of corpus_idx values (arbitrary-order support).
   - Selector callback: called each round with (remaining: list[SourceCandidate],
     current_evidence: list[HumanEvidence]) → (corpus_idx, optional_gain).
   - Callback receives only SourceCandidate views (no target_texts).

5. Validation:
   - Selector must return a corpus_idx in remaining candidates.
   - No duplicate reveals.
   - Deterministic replay.

6. Scope: 40 acquisitions → 41 states. Tiny tests cover all invariants.
7. NOT implemented: prediction callbacks, retrieval, outcomes, persistence,
   metrics, plots, CLI/GPU.
"""
from __future__ import annotations

import copy
from typing import Callable

import pytest

from constrained_translation.experiment.sequencing_runner import (
    SourceCandidate,
    HumanEvidence,
    RoundState,
    RoundSchedule,
    PoolValidationError,
    source_candidates_from_manifest,
    human_evidence_from_manifest,
    validate_pools,
    build_round_schedule,
    build_round_schedule_from_selector,
)
from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

LANG = "mya"


def _pool_item(corpus_idx: int, source: str, target: str = "") -> SequencingPoolItem:
    tgt = target if target else f"tgt_{corpus_idx}"
    return SequencingPoolItem(
        corpus_idx=corpus_idx,
        item_id=f"TST {corpus_idx}:1",
        source_text=source,
        target_texts={LANG: tgt},
    )


def _make_manifest(
    seed_items: list[SequencingPoolItem],
    acq_items: list[SequencingPoolItem],
    eval_items: list[SequencingPoolItem],
    remaining_items: list[SequencingPoolItem],
) -> SequencingManifest:
    return SequencingManifest(
        languages=[LANG],
        random_seed=0,
        seed_size=len(seed_items),
        acq_size=len(acq_items),
        eval_size=len(eval_items),
        seed=seed_items,
        acquisition=acq_items,
        fixed_eval=eval_items,
        remaining=remaining_items,
        digest="",
    )


def _unique_sources(prefix: str, n: int, start: int = 0) -> list[str]:
    """Return n unique source texts with distinct normalized forms."""
    phrases = [
        "the lord said unto moses",
        "in the beginning god created",
        "blessed are the poor in spirit",
        "god saw that it was good",
        "and the spirit moved upon the waters",
        "let there be light and there was light",
        "love your neighbor as yourself",
        "thou shalt not kill or steal",
        "the earth was without form and void",
        "all the days of his life",
        "he arose and went to his father",
        "and they lived happily ever after",
        "the word of god is truth",
        "seek and you shall find it",
        "ask and it shall be given",
    ]
    return [f"{prefix} {phrases[(start + i) % len(phrases)]} number {start + i}" for i in range(n)]


@pytest.fixture
def tiny_manifest():
    """A minimal valid manifest with 3 seed, 4 acq, 2 eval, 5 remaining items."""
    seed_srcs = _unique_sources("seed", 3, 0)
    acq_srcs = _unique_sources("acq", 4, 3)
    eval_srcs = _unique_sources("eval", 2, 7)
    rem_srcs = _unique_sources("rem", 5, 9)

    seed_items = [_pool_item(i, seed_srcs[i], f"seed_tgt_{i}") for i in range(3)]
    acq_items = [_pool_item(10 + i, acq_srcs[i], f"acq_tgt_{i}") for i in range(4)]
    eval_items = [_pool_item(20 + i, eval_srcs[i], f"eval_tgt_{i}") for i in range(2)]
    rem_items = [_pool_item(30 + i, rem_srcs[i]) for i in range(5)]

    return _make_manifest(seed_items, acq_items, eval_items, rem_items)


# ---------------------------------------------------------------------------
# Section 1: Data model tests
# ---------------------------------------------------------------------------

class TestSourceCandidate:
    """SourceCandidate is a leakage-safe, source-only view."""

    def test_has_corpus_idx_item_id_source_text(self):
        sc = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        assert sc.corpus_idx == 5
        assert sc.item_id == "GEN 1:5"
        assert sc.source_text == "and god said"

    def test_no_target_text_field(self):
        sc = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        assert not hasattr(sc, "target_text")
        assert not hasattr(sc, "target_texts")

    def test_is_immutable(self):
        sc = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        with pytest.raises((AttributeError, TypeError)):
            sc.corpus_idx = 99  # type: ignore[misc]

    def test_equality_by_corpus_idx(self):
        a = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        b = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        assert a == b

    def test_hashable(self):
        sc = SourceCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="and god said")
        s = {sc}
        assert sc in s


class TestHumanEvidence:
    """HumanEvidence carries source + target for a revealed item."""

    def test_has_corpus_idx_item_id_source_text_target_text(self):
        he = HumanEvidence(corpus_idx=3, item_id="GEN 1:3", source_text="let there be light", target_text="target_here")
        assert he.corpus_idx == 3
        assert he.item_id == "GEN 1:3"
        assert he.source_text == "let there be light"
        assert he.target_text == "target_here"

    def test_target_text_nonempty_enforced(self):
        with pytest.raises((ValueError, TypeError)):
            HumanEvidence(corpus_idx=1, item_id="X", source_text="src", target_text="")

    def test_is_immutable(self):
        he = HumanEvidence(corpus_idx=3, item_id="GEN 1:3", source_text="light", target_text="lux")
        with pytest.raises((AttributeError, TypeError)):
            he.target_text = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 2: Adapter functions
# ---------------------------------------------------------------------------

class TestAdapters:
    def test_source_candidates_from_manifest_strips_targets(self, tiny_manifest):
        candidates = source_candidates_from_manifest(tiny_manifest)
        assert len(candidates) == len(tiny_manifest.acquisition)
        for sc in candidates:
            assert isinstance(sc, SourceCandidate)
            assert not hasattr(sc, "target_text")
            assert not hasattr(sc, "target_texts")

    def test_source_candidates_corpus_indices_match(self, tiny_manifest):
        candidates = source_candidates_from_manifest(tiny_manifest)
        expected_idxs = [item.corpus_idx for item in tiny_manifest.acquisition]
        assert [sc.corpus_idx for sc in candidates] == expected_idxs

    def test_source_candidates_source_texts_match(self, tiny_manifest):
        candidates = source_candidates_from_manifest(tiny_manifest)
        for sc, item in zip(candidates, tiny_manifest.acquisition):
            assert sc.source_text == item.source_text

    def test_human_evidence_from_manifest_seed(self, tiny_manifest):
        evidence = human_evidence_from_manifest(tiny_manifest, LANG, pool="seed")
        assert len(evidence) == len(tiny_manifest.seed)
        for he, item in zip(evidence, tiny_manifest.seed):
            assert isinstance(he, HumanEvidence)
            assert he.corpus_idx == item.corpus_idx
            assert he.source_text == item.source_text
            assert he.target_text == item.target_texts[LANG]

    def test_human_evidence_from_manifest_acquisition(self, tiny_manifest):
        evidence = human_evidence_from_manifest(tiny_manifest, LANG, pool="acquisition")
        assert len(evidence) == len(tiny_manifest.acquisition)
        for he, item in zip(evidence, tiny_manifest.acquisition):
            assert he.target_text == item.target_texts[LANG]

    def test_human_evidence_from_manifest_nonempty_target(self, tiny_manifest):
        evidence = human_evidence_from_manifest(tiny_manifest, LANG, pool="seed")
        for he in evidence:
            assert he.target_text  # nonempty

    def test_human_evidence_unknown_pool_raises(self, tiny_manifest):
        with pytest.raises((ValueError, KeyError)):
            human_evidence_from_manifest(tiny_manifest, LANG, pool="nonexistent_pool")


# ---------------------------------------------------------------------------
# Section 3: Pool validation
# ---------------------------------------------------------------------------

class TestValidatePools:
    def test_valid_tiny_manifest_passes(self, tiny_manifest):
        # Should not raise
        validate_pools(tiny_manifest, LANG,
                       expected_seed=3, expected_acq=4,
                       expected_eval=2, expected_project_min=1)

    def test_wrong_seed_cardinality_raises(self, tiny_manifest):
        with pytest.raises(PoolValidationError):
            validate_pools(tiny_manifest, LANG,
                           expected_seed=99, expected_acq=4,
                           expected_eval=2, expected_project_min=1)

    def test_wrong_acq_cardinality_raises(self, tiny_manifest):
        with pytest.raises(PoolValidationError):
            validate_pools(tiny_manifest, LANG,
                           expected_seed=3, expected_acq=99,
                           expected_eval=2, expected_project_min=1)

    def test_wrong_eval_cardinality_raises(self, tiny_manifest):
        with pytest.raises(PoolValidationError):
            validate_pools(tiny_manifest, LANG,
                           expected_seed=3, expected_acq=4,
                           expected_eval=99, expected_project_min=1)

    def test_insufficient_project_size_raises(self, tiny_manifest):
        with pytest.raises(PoolValidationError):
            validate_pools(tiny_manifest, LANG,
                           expected_seed=3, expected_acq=4,
                           expected_eval=2, expected_project_min=999)

    def test_unique_corpus_indices_in_seed(self):
        # Duplicate corpus_idx in seed → raise
        dup_items = [
            _pool_item(1, "source alpha"),
            _pool_item(1, "source alpha"),  # duplicate!
        ]
        m = _make_manifest(dup_items, [], [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=2, expected_acq=0,
                           expected_eval=0, expected_project_min=0)

    def test_unique_corpus_indices_in_acq(self):
        dup_items = [
            _pool_item(10, "source one"),
            _pool_item(10, "source one"),  # duplicate!
        ]
        m = _make_manifest([], dup_items, [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=0, expected_acq=2,
                           expected_eval=0, expected_project_min=0)

    def test_no_seed_acq_overlap(self):
        seed_items = [_pool_item(5, "shared source text here")]
        acq_items = [_pool_item(5, "shared source text here")]  # same corpus_idx!
        m = _make_manifest(seed_items, acq_items, [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=1, expected_acq=1,
                           expected_eval=0, expected_project_min=0)

    def test_nonempty_target_in_seed(self):
        # Item with empty target in seed → raise
        bad = SequencingPoolItem(corpus_idx=1, item_id="X", source_text="src",
                                 target_texts={LANG: ""})
        m = _make_manifest([bad], [], [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=1, expected_acq=0,
                           expected_eval=0, expected_project_min=0)

    def test_nonempty_target_in_acq(self):
        bad = SequencingPoolItem(corpus_idx=10, item_id="X", source_text="src",
                                 target_texts={LANG: ""})
        m = _make_manifest([], [bad], [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=0, expected_acq=1,
                           expected_eval=0, expected_project_min=0)

    def test_normalized_source_equivalent_leakage_seed_acq(self):
        """Seed and acquisition may not share normalized-source equivalent."""
        # Both items normalize to same key
        seed_items = [_pool_item(1, "god said let there be light")]
        acq_items = [_pool_item(2, "God said let there be light")]  # same norm key
        m = _make_manifest(seed_items, acq_items, [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=1, expected_acq=1,
                           expected_eval=0, expected_project_min=0)

    def test_normalized_source_equivalent_leakage_seed_eval(self):
        """Seed and fixed_eval may not share normalized-source equivalent."""
        seed_items = [_pool_item(1, "god said let there be light")]
        eval_items = [_pool_item(3, "God said let there be light")]
        m = _make_manifest(seed_items, [], eval_items, [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=1, expected_acq=0,
                           expected_eval=1, expected_project_min=0)

    def test_normalized_source_equivalent_leakage_acq_eval(self):
        """Acquisition and fixed_eval may not share normalized-source equivalent."""
        acq_items = [_pool_item(2, "god said let there be light")]
        eval_items = [_pool_item(3, "God said let there be light")]
        m = _make_manifest([], acq_items, eval_items, [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=0, expected_acq=1,
                           expected_eval=1, expected_project_min=0)

    def test_project_may_contain_equivalents(self, tiny_manifest):
        """Remaining/project pool may have items with same norm key (not a violation)."""
        # Add a duplicate norm-key item to remaining
        extra = _pool_item(100, tiny_manifest.remaining[0].source_text)
        m = _make_manifest(
            tiny_manifest.seed,
            tiny_manifest.acquisition,
            tiny_manifest.fixed_eval,
            tiny_manifest.remaining + [extra],
        )
        # Should NOT raise
        validate_pools(m, LANG, expected_seed=3, expected_acq=4,
                       expected_eval=2, expected_project_min=1)

    def test_missing_language_in_seed_raises(self):
        # Item lacks the requested language in target_texts
        bad = SequencingPoolItem(corpus_idx=1, item_id="X", source_text="src",
                                 target_texts={"ckb": "other"})
        m = _make_manifest([bad], [], [], [])
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG, expected_seed=1, expected_acq=0,
                           expected_eval=0, expected_project_min=0)


# ---------------------------------------------------------------------------
# Section 4: RoundState immutability and structure
# ---------------------------------------------------------------------------

class TestRoundState:
    def test_round_0_structure(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],  # corpus_idx of acq items
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[0]
        assert state.round == 0
        assert len(state.evidence) == 3  # seed only
        assert state.selected is None  # nothing selected yet
        assert state.estimated_gain is None
        # Remaining candidates = all 4 acquisition items
        assert len(state.remaining_candidates) == 4

    def test_round_0_evidence_is_seed_only(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[0]
        seed_idxs = {item.corpus_idx for item in tiny_manifest.seed}
        evidence_idxs = {e.corpus_idx for e in state.evidence}
        assert evidence_idxs == seed_idxs

    def test_round_1_adds_one_acquisition(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[1]
        assert state.round == 1
        assert len(state.evidence) == 4  # seed + 1
        assert state.selected is not None
        assert state.selected.corpus_idx == 10  # first in order

    def test_evidence_cardinality_seed_plus_n(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for r, state in enumerate(schedule.states):
            assert len(state.evidence) == 3 + r, (
                f"Round {r}: expected {3 + r} evidence items, got {len(state.evidence)}"
            )

    def test_exactly_41_states_for_4_acquisitions(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert len(schedule.states) == 5  # round 0 + 4 acquisitions

    def test_remaining_candidates_decrease_monotonically(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        prev_count = len(schedule.states[0].remaining_candidates)
        for state in schedule.states[1:]:
            assert len(state.remaining_candidates) == prev_count - 1
            prev_count = len(state.remaining_candidates)

    def test_final_round_no_remaining_candidates(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert len(schedule.states[-1].remaining_candidates) == 0

    def test_round_state_is_immutable(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[0]
        with pytest.raises((AttributeError, TypeError)):
            state.round = 99  # type: ignore[misc]

    def test_evidence_is_immutable_sequence(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[0]
        with pytest.raises((AttributeError, TypeError)):
            state.evidence = []  # type: ignore[misc]

    def test_remaining_candidates_is_immutable(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state = schedule.states[0]
        with pytest.raises((AttributeError, TypeError)):
            state.remaining_candidates = []  # type: ignore[misc]

    def test_round_schedule_is_immutable(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            schedule.states = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 5: Target leakage prohibition in policy view
# ---------------------------------------------------------------------------

class TestNoTargetLeakage:
    """No future target data must appear in remaining_candidates (SourceCandidate)."""

    def test_remaining_candidates_are_source_candidates(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states:
            for sc in state.remaining_candidates:
                assert isinstance(sc, SourceCandidate)
                assert not hasattr(sc, "target_text")
                assert not hasattr(sc, "target_texts")

    def test_evidence_items_are_human_evidence(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states:
            for he in state.evidence:
                assert isinstance(he, HumanEvidence)
                assert he.target_text  # has target, is a revealed item

    def test_future_acquisition_targets_not_in_round_0_remaining(self, tiny_manifest):
        """Policy spy: round 0 remaining candidates must NOT expose target_text."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        state0 = schedule.states[0]
        # All acquisition items should be in remaining (round 0), but as SourceCandidate
        acq_idxs = {item.corpus_idx for item in tiny_manifest.acquisition}
        remaining_idxs = {sc.corpus_idx for sc in state0.remaining_candidates}
        assert remaining_idxs == acq_idxs

        # None should have target_text
        for sc in state0.remaining_candidates:
            assert not hasattr(sc, "target_text")
            assert not hasattr(sc, "target_texts")

    def test_selected_in_state_has_no_target(self, tiny_manifest):
        """The 'selected' metadata field should be a SourceCandidate, not HumanEvidence."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states[1:]:
            assert isinstance(state.selected, SourceCandidate)
            assert not hasattr(state.selected, "target_text")

    def test_policy_spy_no_future_targets_in_evidence(self, tiny_manifest):
        """Selector callback must not receive future target info through evidence."""
        seen_future_targets: list[str] = []
        acq_targets = {item.corpus_idx: item.target_texts[LANG]
                       for item in tiny_manifest.acquisition}

        call_count = [0]
        def spy_selector(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            call_count[0] += 1
            # Check that no remaining candidate has a target attribute
            for sc in remaining:
                if hasattr(sc, "target_text"):
                    seen_future_targets.append(f"round_spy: candidate {sc.corpus_idx} has target_text")
            # Check that evidence doesn't include future (not-yet-revealed) targets
            revealed_idxs = {he.corpus_idx for he in evidence}
            # The selector sees only source-side data for remaining candidates
            for sc in remaining:
                if sc.corpus_idx in revealed_idxs:
                    seen_future_targets.append(f"candidate {sc.corpus_idx} already in evidence?")
            # Select first remaining by corpus_idx
            chosen = min(remaining, key=lambda sc: sc.corpus_idx)
            return chosen.corpus_idx, None

        build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=spy_selector,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert call_count[0] == 4  # called once per acquisition round
        assert seen_future_targets == [], f"Leakage detected: {seen_future_targets}"


# ---------------------------------------------------------------------------
# Section 6: Precomputed order input
# ---------------------------------------------------------------------------

class TestPrecomputedOrder:
    def test_acq_order_determines_reveal_sequence(self, tiny_manifest):
        # Specify reverse order
        order = [13, 12, 11, 10]
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=order,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for r, expected_idx in enumerate(order, start=1):
            assert schedule.states[r].selected.corpus_idx == expected_idx

    def test_invalid_corpus_idx_in_order_raises(self, tiny_manifest):
        bad_order = [10, 11, 12, 9999]  # 9999 not in acquisition
        with pytest.raises((ValueError, KeyError, PoolValidationError)):
            build_round_schedule(
                manifest=tiny_manifest,
                language=LANG,
                acq_order=bad_order,
                expected_seed=3, expected_acq=4,
                expected_eval=2, expected_project_min=1,
            )

    def test_duplicate_in_order_raises(self, tiny_manifest):
        dup_order = [10, 10, 12, 13]  # 10 appears twice
        with pytest.raises((ValueError, PoolValidationError)):
            build_round_schedule(
                manifest=tiny_manifest,
                language=LANG,
                acq_order=dup_order,
                expected_seed=3, expected_acq=4,
                expected_eval=2, expected_project_min=1,
            )

    def test_incomplete_order_raises(self, tiny_manifest):
        """Order must cover all acquisition items."""
        partial = [10, 11]  # missing 12, 13
        with pytest.raises((ValueError, PoolValidationError)):
            build_round_schedule(
                manifest=tiny_manifest,
                language=LANG,
                acq_order=partial,
                expected_seed=3, expected_acq=4,
                expected_eval=2, expected_project_min=1,
            )

    def test_deterministic_replay(self, tiny_manifest):
        """Same precomputed order → identical schedule."""
        order = [13, 12, 11, 10]
        s1 = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=order,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        s2 = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=order,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert len(s1.states) == len(s2.states)
        for st1, st2 in zip(s1.states, s2.states):
            assert st1.round == st2.round
            assert [e.corpus_idx for e in st1.evidence] == [e.corpus_idx for e in st2.evidence]
            if st1.selected is not None:
                assert st1.selected.corpus_idx == st2.selected.corpus_idx


# ---------------------------------------------------------------------------
# Section 7: Selector callback input
# ---------------------------------------------------------------------------

class TestSelectorCallback:
    def test_selector_receives_remaining_source_candidates(self, tiny_manifest):
        received: list[list[SourceCandidate]] = []

        def sel(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            received.append(list(remaining))
            return min(r.corpus_idx for r in remaining), None

        build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=sel,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert len(received) == 4  # 4 rounds
        assert len(received[0]) == 4  # all 4 available at round 1
        assert len(received[1]) == 3  # 3 after first selection
        assert len(received[2]) == 2
        assert len(received[3]) == 1

    def test_selector_receives_growing_evidence(self, tiny_manifest):
        received_evidence: list[list[HumanEvidence]] = []

        def sel(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            received_evidence.append(list(evidence))
            return min(r.corpus_idx for r in remaining), None

        build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=sel,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        # evidence grows by 1 each round (seed=3 at round 1, seed+1=4 at round 2, etc.)
        # Selector is called at round 1..4; before selection, evidence has seed + (r-1) items
        assert len(received_evidence[0]) == 3  # seed only at round 1
        assert len(received_evidence[1]) == 4  # seed + 1
        assert len(received_evidence[2]) == 5  # seed + 2
        assert len(received_evidence[3]) == 6  # seed + 3

    def test_selector_invalid_choice_raises(self, tiny_manifest):
        """Selector returning corpus_idx not in remaining → ValueError."""
        def bad_selector(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            return 9999, None  # not a valid corpus_idx

        with pytest.raises(ValueError):
            build_round_schedule_from_selector(
                manifest=tiny_manifest,
                language=LANG,
                selector=bad_selector,
                expected_seed=3, expected_acq=4,
                expected_eval=2, expected_project_min=1,
            )

    def test_selector_returns_estimated_gain(self, tiny_manifest):
        """Selector can return an estimated gain value that is stored in state."""
        gains = [1.5, 2.0, 0.8, 3.1]
        call = [0]

        def sel_with_gain(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, float]:
            gain = gains[call[0]]
            call[0] += 1
            return min(r.corpus_idx for r in remaining), gain

        schedule = build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=sel_with_gain,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for r, expected_gain in enumerate(gains, start=1):
            assert schedule.states[r].estimated_gain == expected_gain

    def test_selector_deterministic_replay(self, tiny_manifest):
        """Same selector logic → identical schedule on two calls."""
        def det_sel(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            return min(r.corpus_idx for r in remaining), None

        s1 = build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=det_sel,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        s2 = build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=det_sel,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for st1, st2 in zip(s1.states, s2.states):
            assert [e.corpus_idx for e in st1.evidence] == [e.corpus_idx for e in st2.evidence]

    def test_no_duplicate_reveal(self, tiny_manifest):
        """Each corpus_idx revealed at most once across all rounds."""
        revealed: list[int] = []
        def sel(remaining: list[SourceCandidate], evidence: list[HumanEvidence]) -> tuple[int, None]:
            chosen = min(r.corpus_idx for r in remaining)
            return chosen, None

        schedule = build_round_schedule_from_selector(
            manifest=tiny_manifest,
            language=LANG,
            selector=sel,
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states[1:]:
            assert state.selected.corpus_idx not in revealed
            revealed.append(state.selected.corpus_idx)


# ---------------------------------------------------------------------------
# Section 8: Evidence ID and count monotonicity
# ---------------------------------------------------------------------------

class TestEvidenceMonotonicity:
    def test_evidence_corpus_indices_monotonically_grow(self, tiny_manifest):
        """Each round's evidence is a superset of the previous round's."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        prev_idxs: set[int] = set()
        for state in schedule.states:
            curr_idxs = {e.corpus_idx for e in state.evidence}
            assert prev_idxs.issubset(curr_idxs), (
                f"Round {state.round}: evidence shrunk — prev {prev_idxs}, curr {curr_idxs}"
            )
            prev_idxs = curr_idxs

    def test_evidence_count_increases_by_one_each_round(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for i in range(1, len(schedule.states)):
            prev = len(schedule.states[i - 1].evidence)
            curr = len(schedule.states[i].evidence)
            assert curr == prev + 1

    def test_evidence_ids_unique_within_state(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states:
            idxs = [e.corpus_idx for e in state.evidence]
            assert len(idxs) == len(set(idxs)), (
                f"Round {state.round}: duplicate corpus_idx in evidence: {idxs}"
            )

    def test_one_target_revealed_per_round(self, tiny_manifest):
        """Each round reveals exactly one new target (via selected)."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for state in schedule.states[1:]:
            assert state.selected is not None
            # The selected item should now appear in evidence
            revealed_idxs = {e.corpus_idx for e in state.evidence}
            assert state.selected.corpus_idx in revealed_idxs


# ---------------------------------------------------------------------------
# Section 9: Schedule-level metadata
# ---------------------------------------------------------------------------

class TestScheduleMetadata:
    def test_schedule_stores_manifest_reference(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert schedule.manifest is tiny_manifest

    def test_schedule_stores_language(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert schedule.language == LANG

    def test_schedule_round_indices_sequential(self, tiny_manifest):
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        for i, state in enumerate(schedule.states):
            assert state.round == i

    def test_schedule_states_tuple_not_list(self, tiny_manifest):
        """States should be immutable sequence (tuple)."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        assert isinstance(schedule.states, tuple)


# ---------------------------------------------------------------------------
# Section 10: Production defaults validation (abstract — no corpus needed)
# ---------------------------------------------------------------------------

class TestProductionDefaults:
    """Verify that validate_pools enforces exact production cardinalities
    when expected_* are left at their defaults."""

    def _make_production_manifest(self) -> SequencingManifest:
        """Build a manifest that matches production defaults."""
        seed_items = [_pool_item(i, f"seed source sentence {i} long enough") for i in range(10)]
        acq_items = [_pool_item(100 + i, f"acq source sentence {100 + i} long enough") for i in range(40)]
        eval_items = [_pool_item(200 + i, f"eval source sentence {200 + i} long enough") for i in range(60)]
        rem_items = [_pool_item(400 + i, f"rem source sentence {400 + i} long enough") for i in range(200)]
        return _make_manifest(seed_items, acq_items, eval_items, rem_items)

    def test_production_defaults_pass(self):
        m = self._make_production_manifest()
        # Default production: seed=10, acq=40, eval=60, project_min=200
        validate_pools(m, LANG,
                       expected_seed=10, expected_acq=40,
                       expected_eval=60, expected_project_min=200)

    def test_wrong_seed_fails_production(self):
        m = self._make_production_manifest()
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG,
                           expected_seed=9, expected_acq=40,
                           expected_eval=60, expected_project_min=200)

    def test_wrong_acq_fails_production(self):
        m = self._make_production_manifest()
        with pytest.raises(PoolValidationError):
            validate_pools(m, LANG,
                           expected_seed=10, expected_acq=39,
                           expected_eval=60, expected_project_min=200)


# ---------------------------------------------------------------------------
# Section 11: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_evidence_tuple_in_round_0_if_no_seed(self):
        """If seed is empty (unusual), round 0 still produces a valid state."""
        acq_items = [_pool_item(10, "acq source sentence one")]
        m = _make_manifest([], acq_items, [], [])
        # Can't validate with expected_seed=0... let's just build directly
        # Without pool validation cardinality enforcement for 0-seed edge case
        schedule = build_round_schedule(
            manifest=m,
            language=LANG,
            acq_order=[10],
            expected_seed=0, expected_acq=1,
            expected_eval=0, expected_project_min=0,
        )
        assert schedule.states[0].round == 0
        assert len(schedule.states[0].evidence) == 0
        assert len(schedule.states) == 2  # round 0 + 1 acquisition

    def test_single_acquisition_gives_2_states(self):
        seed_items = [_pool_item(0, "seed source text one")]
        acq_items = [_pool_item(10, "acq source text two")]
        m = _make_manifest(seed_items, acq_items, [], [])
        schedule = build_round_schedule(
            manifest=m,
            language=LANG,
            acq_order=[10],
            expected_seed=1, expected_acq=1,
            expected_eval=0, expected_project_min=0,
        )
        assert len(schedule.states) == 2
        assert schedule.states[0].round == 0
        assert schedule.states[1].round == 1

    def test_evidence_items_preserve_source_texts(self, tiny_manifest):
        """Evidence HumanEvidence.source_text must match manifest item."""
        schedule = build_round_schedule(
            manifest=tiny_manifest,
            language=LANG,
            acq_order=[10, 11, 12, 13],
            expected_seed=3, expected_acq=4,
            expected_eval=2, expected_project_min=1,
        )
        # Build lookup
        all_manifest_items = {
            item.corpus_idx: item
            for item in tiny_manifest.seed + tiny_manifest.acquisition
        }
        for state in schedule.states:
            for he in state.evidence:
                original = all_manifest_items[he.corpus_idx]
                assert he.source_text == original.source_text
                assert he.target_text == original.target_texts[LANG]
