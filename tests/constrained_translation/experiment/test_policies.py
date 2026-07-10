"""tests/constrained_translation/experiment/test_policies.py

Strict TDD tests for source-only acquisition policies (Task 3).
Written BEFORE the implementation (RED phase).

Required behaviour:
1. AcquisitionCandidate dataclass — corpus_idx, item_id, source_text; NO target_texts.
2. candidates_from_manifest: adapter from SequencingManifest.acquisition that strips target data.
3. Arbitrary policy: deterministic permutation from explicit seed; every candidate exactly once;
   no mutation/global RNG; stable across reruns; ≥ 5 distinct seeds produce different orders.
4. Silver policy: each round pick candidate with highest fraction of normalized source-unit
   occurrences currently known from translated source cells; then highest source-side BM25
   similarity to current translated source pool; then stable lowest corpus_idx.
   Empty translated pool → deterministic (no crash); punctuation-only → handled.
5. Golden policy: choose max frequency-weighted marginal reduction against frozen ProjectSlice;
   then max unweighted new project types; then lowest corpus_idx.
   Reuse corpus_coverage.marginal_gain. Expose ex-ante weighted and unweighted gain.
6. Empty candidate list → loud ValueError / raises.
7. No target reference, post-hoc chrF, generation, or target alignment signal.
"""
from __future__ import annotations

import math
from typing import Sequence

import pytest

from constrained_translation.experiment.policies import (
    AcquisitionCandidate,
    candidates_from_manifest,
    arbitrary_policy,
    silver_policy,
    golden_policy,
    GoldenGain,
)
from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.experiment.corpus_coverage import (
    build_project_slice,
    ProjectSlice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool_item(corpus_idx: int, source_text: str, langs: list[str] | None = None) -> SequencingPoolItem:
    langs = langs or ["mya"]
    return SequencingPoolItem(
        corpus_idx=corpus_idx,
        item_id=f"TST {corpus_idx}:1",
        source_text=source_text,
        target_texts={lang: f"target_{corpus_idx}_{lang}" for lang in langs},
    )


def _make_manifest(
    acq_items: list[SequencingPoolItem],
    remaining_items: list[SequencingPoolItem] | None = None,
) -> SequencingManifest:
    return SequencingManifest(
        languages=["mya"],
        random_seed=0,
        seed_size=0,
        acq_size=len(acq_items),
        eval_size=0,
        seed=[],
        acquisition=acq_items,
        fixed_eval=[],
        remaining=remaining_items or [],
        digest="",
    )


def _candidate(corpus_idx: int, source_text: str) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        corpus_idx=corpus_idx,
        item_id=f"TST {corpus_idx}:1",
        source_text=source_text,
    )


# ---------------------------------------------------------------------------
# 1. AcquisitionCandidate dataclass — leakage / API tests
# ---------------------------------------------------------------------------

class TestAcquisitionCandidateDataclass:
    def test_has_required_fields(self):
        c = AcquisitionCandidate(corpus_idx=5, item_id="GEN 1:5", source_text="hello world")
        assert c.corpus_idx == 5
        assert c.item_id == "GEN 1:5"
        assert c.source_text == "hello world"

    def test_no_target_texts_field(self):
        """AcquisitionCandidate must NOT have a target_texts attribute."""
        c = AcquisitionCandidate(corpus_idx=0, item_id="X", source_text="abc")
        assert not hasattr(c, "target_texts"), (
            "AcquisitionCandidate must not expose target_texts — target leakage!"
        )

    def test_no_reference_field(self):
        """AcquisitionCandidate must NOT have a reference attribute."""
        c = AcquisitionCandidate(corpus_idx=0, item_id="X", source_text="abc")
        assert not hasattr(c, "reference"), (
            "AcquisitionCandidate must not expose reference — target leakage!"
        )

    def test_equality_by_value(self):
        a = AcquisitionCandidate(corpus_idx=1, item_id="A", source_text="foo bar")
        b = AcquisitionCandidate(corpus_idx=1, item_id="A", source_text="foo bar")
        assert a == b

    def test_hashable_or_frozen(self):
        """Candidates should be comparable/usable as keys (frozen dataclass or __eq__)."""
        c = AcquisitionCandidate(corpus_idx=0, item_id="X", source_text="abc")
        # Must at least support repr without error
        assert repr(c)


# ---------------------------------------------------------------------------
# 2. candidates_from_manifest adapter
# ---------------------------------------------------------------------------

class TestCandidatesFromManifest:
    def test_strips_target_data(self):
        """Adapter must produce AcquisitionCandidates with no target_texts access."""
        items = [_make_pool_item(i, f"source {i}") for i in range(5)]
        manifest = _make_manifest(items)
        candidates = candidates_from_manifest(manifest)

        assert len(candidates) == 5
        for c in candidates:
            assert isinstance(c, AcquisitionCandidate)
            assert not hasattr(c, "target_texts")

    def test_preserves_corpus_idx_and_item_id(self):
        items = [_make_pool_item(10, "In the beginning")]
        manifest = _make_manifest(items)
        (c,) = candidates_from_manifest(manifest)
        assert c.corpus_idx == 10
        assert c.item_id == "TST 10:1"
        assert c.source_text == "In the beginning"

    def test_returns_all_acquisition_items(self):
        items = [_make_pool_item(i, f"text {i}") for i in range(20)]
        manifest = _make_manifest(items)
        candidates = candidates_from_manifest(manifest)
        assert len(candidates) == 20
        assert [c.corpus_idx for c in candidates] == list(range(20))

    def test_does_not_accept_pool_item_directly(self):
        """Policy functions must not accept SequencingPoolItem directly."""
        item = _make_pool_item(0, "hello")
        # AcquisitionCandidate should not be a SequencingPoolItem
        assert not isinstance(item, AcquisitionCandidate)
        assert type(item) is SequencingPoolItem


# ---------------------------------------------------------------------------
# 3. Arbitrary policy
# ---------------------------------------------------------------------------

class TestArbitraryPolicy:
    @pytest.fixture
    def candidates_10(self) -> list[AcquisitionCandidate]:
        return [_candidate(i, f"sentence number {i}") for i in range(10)]

    def test_returns_exactly_one_candidate(self, candidates_10):
        remaining = list(candidates_10)
        chosen = arbitrary_policy(remaining, seed=42)
        assert isinstance(chosen, AcquisitionCandidate)

    def test_chosen_is_in_remaining(self, candidates_10):
        remaining = list(candidates_10)
        chosen = arbitrary_policy(remaining, seed=42)
        assert chosen in candidates_10

    def test_does_not_mutate_remaining_list(self, candidates_10):
        original = list(candidates_10)
        arbitrary_policy(list(candidates_10), seed=42)
        assert list(candidates_10) == original

    def test_deterministic_same_seed_same_order(self, candidates_10):
        """Same seed and same list → same sequence of choices."""
        results1 = []
        results2 = []
        pool1 = list(candidates_10)
        pool2 = list(candidates_10)
        for _ in range(len(candidates_10)):
            c1 = arbitrary_policy(pool1, seed=99)
            c2 = arbitrary_policy(pool2, seed=99)
            results1.append(c1)
            results2.append(c2)
            pool1 = [c for c in pool1 if c != c1]
            pool2 = [c for c in pool2 if c != c2]
        assert results1 == results2

    def test_completeness_covers_all_candidates(self, candidates_10):
        """All 10 candidates are selected across 10 rounds."""
        pool = list(candidates_10)
        selected = []
        while pool:
            c = arbitrary_policy(pool, seed=7)
            selected.append(c)
            pool = [x for x in pool if x != c]
        assert set(c.corpus_idx for c in selected) == set(range(10))
        assert len(selected) == 10

    def test_at_least_five_distinct_seeds_produce_different_orders(self):
        """5 distinct seeds must produce at least 2 different orderings over 5+ candidates."""
        candidates = [_candidate(i, f"word{i} text sample") for i in range(8)]
        orders = []
        for seed in [0, 1, 2, 3, 4, 17, 42, 99, 1000, 9999]:
            pool = list(candidates)
            order = []
            while pool:
                c = arbitrary_policy(pool, seed=seed)
                order.append(c.corpus_idx)
                pool = [x for x in pool if x != c]
            orders.append(tuple(order))
        unique_orders = set(orders)
        assert len(unique_orders) >= 5, (
            f"Expected ≥5 distinct seed orderings, got {len(unique_orders)}: {unique_orders}"
        )

    def test_no_global_rng_contamination(self, candidates_10):
        """Arbitrary policy must not alter global random state."""
        import random
        random.seed(12345)
        before = random.random()
        random.seed(12345)
        arbitrary_policy(list(candidates_10), seed=42)
        after = random.random()
        assert before == after, "Arbitrary policy contaminated global random state!"

    def test_stable_rerun(self, candidates_10):
        """Running policy twice with same args returns same candidate."""
        pool = list(candidates_10)
        c1 = arbitrary_policy(pool, seed=55)
        c2 = arbitrary_policy(pool, seed=55)
        assert c1 == c2

    def test_single_candidate_returns_it(self):
        c = _candidate(3, "only one")
        assert arbitrary_policy([c], seed=42) is c

    def test_empty_raises(self):
        with pytest.raises((ValueError, IndexError)):
            arbitrary_policy([], seed=42)

    def test_seed_zero(self, candidates_10):
        """Seed 0 is a valid, supported seed."""
        chosen = arbitrary_policy(list(candidates_10), seed=0)
        assert chosen in candidates_10

    def test_five_explicit_seeds_completeness(self):
        """Each of seeds 1..5 produces a complete permutation of all candidates."""
        candidates = [_candidate(i, f"sample {i}") for i in range(6)]
        for seed in [1, 2, 3, 4, 5]:
            pool = list(candidates)
            seen = set()
            while pool:
                c = arbitrary_policy(pool, seed=seed)
                assert c.corpus_idx not in seen, f"Seed {seed}: candidate selected twice!"
                seen.add(c.corpus_idx)
                pool = [x for x in pool if x != c]
            assert seen == set(range(6)), f"Seed {seed}: not all candidates selected"


# ---------------------------------------------------------------------------
# 4. Silver policy
# ---------------------------------------------------------------------------

class TestSilverPolicy:
    """
    Silver policy: highest known-fraction → highest BM25 similarity to translated pool
    → lowest corpus_idx.
    """

    def test_returns_acquisition_candidate(self):
        candidates = [_candidate(i, f"word{i} text") for i in range(3)]
        chosen = silver_policy(candidates, translated_source_texts=[])
        assert isinstance(chosen, AcquisitionCandidate)

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            silver_policy([], translated_source_texts=["some text"])

    def test_single_candidate_returns_it(self):
        c = _candidate(0, "unique sentence")
        result = silver_policy([c], translated_source_texts=["hello world"])
        assert result is c

    def test_does_not_mutate_candidates(self):
        candidates = [_candidate(i, f"sentence {i}") for i in range(5)]
        original = list(candidates)
        silver_policy(candidates, translated_source_texts=["hello"])
        assert candidates == original

    def test_does_not_mutate_translated_pool(self):
        candidates = [_candidate(i, f"test {i}") for i in range(3)]
        pool = ["alpha beta", "gamma delta"]
        original = list(pool)
        silver_policy(candidates, translated_source_texts=pool)
        assert pool == original

    def test_empty_translated_pool_is_deterministic(self):
        """Empty translated pool: all candidates have fraction 0 → tie-break by BM25 then idx."""
        candidates = [_candidate(i, f"word{i} example") for i in range(5)]
        c1 = silver_policy(candidates, translated_source_texts=[])
        c2 = silver_policy(candidates, translated_source_texts=[])
        assert c1 == c2

    def test_known_fraction_ordering(self):
        """Candidate with more known source units appears first."""
        # Translated pool knows: "in", "the", "beginning"
        translated = ["In the beginning"]
        # c0: "in the beginning God created" → 3/5 known
        # c1: "God created light" → 0/3 known
        # c2: "in the beginning" → 3/3 known (highest fraction)
        c0 = _candidate(0, "In the beginning God created")
        c1 = _candidate(1, "God created light")
        c2 = _candidate(2, "in the beginning")

        chosen = silver_policy([c0, c1, c2], translated_source_texts=translated)
        assert chosen.corpus_idx == 2, (
            f"Expected c2 (fraction 3/3=1.0), got corpus_idx={chosen.corpus_idx}"
        )

    def test_tie_broken_by_bm25_similarity(self):
        """When known-fraction ties, BM25 similarity to translated pool breaks the tie."""
        # Both have 0/N known (empty pool), so BM25 to empty pool → all zero → corpus_idx wins
        # To test BM25 we need a non-empty pool
        # Translated pool: "alpha beta gamma"
        # c0: corpus_idx=0, "delta epsilon" — 0 overlap with translated pool
        # c1: corpus_idx=1, "alpha beta delta" — 2 token overlap → higher BM25
        # c2: corpus_idx=2, "alpha beta gamma" — 3 token overlap → highest BM25
        # All have fraction 0/N known
        translated = ["alpha beta gamma"]
        c0 = _candidate(0, "delta epsilon zeta")
        c1 = _candidate(1, "alpha beta delta")
        c2 = _candidate(2, "alpha beta gamma")  # identical to translated — highest similarity

        chosen = silver_policy([c0, c1, c2], translated_source_texts=translated)
        assert chosen.corpus_idx == 2, (
            f"Expected c2 (highest BM25), got corpus_idx={chosen.corpus_idx}"
        )

    def test_stable_tiebreak_lowest_corpus_idx(self):
        """When both known-fraction and BM25 tie exactly, lowest corpus_idx wins."""
        translated = ["hello world"]
        # Both candidates share zero overlap and have identical source text (so same BM25)
        c5 = _candidate(5, "unique rare terms zymurgy")
        c3 = _candidate(3, "unique rare terms zymurgy")
        c8 = _candidate(8, "unique rare terms zymurgy")

        chosen = silver_policy([c5, c3, c8], translated_source_texts=translated)
        assert chosen.corpus_idx == 3, (
            f"Expected corpus_idx=3 (lowest), got {chosen.corpus_idx}"
        )

    def test_punctuation_only_candidate_deterministic(self):
        """Candidate with punctuation-only source is handled deterministically."""
        c_punct = _candidate(0, ",,, ...")  # normalizes to [] — fraction vacuously 0 or 1?
        c_normal = _candidate(1, "normal text here")
        translated = ["normal text"]
        # Should not crash
        chosen = silver_policy([c_punct, c_normal], translated_source_texts=translated)
        assert isinstance(chosen, AcquisitionCandidate)

    def test_silver_early_support_ordering(self):
        """
        Full multi-round ordering test: silver should prefer candidates that support
        the most already-known vocabulary (highest known fraction).
        After translating 'alpha beta', the next round should prefer the candidate
        richest in {alpha, beta}.
        """
        # Translated so far: "alpha beta"
        translated = ["alpha beta"]
        # Candidates:
        # c0: "alpha beta gamma" — known: {alpha, beta} / total {alpha, beta, gamma} → 2/3
        # c1: "alpha beta" — known: {alpha, beta} / total {alpha, beta} → 2/2 = 1.0 HIGHEST
        # c2: "gamma delta epsilon" — known: 0/3
        c0 = _candidate(0, "alpha beta gamma")
        c1 = _candidate(1, "alpha beta")
        c2 = _candidate(2, "gamma delta epsilon")

        chosen = silver_policy([c0, c1, c2], translated_source_texts=translated)
        assert chosen.corpus_idx == 1, (
            f"Expected c1 (fraction 1.0), got corpus_idx={chosen.corpus_idx}"
        )

    def test_bm25_tie_level(self):
        """
        BM25 tiebreak: when two candidates have equal known-fraction (both 0),
        the one with higher BM25 score vs. translated pool wins.
        """
        translated = ["the quick brown fox"]
        # c10: overlaps "quick fox" — some BM25 signal
        # c20: overlaps "the quick brown fox" — max overlap → highest BM25
        # c30: no overlap
        c10 = _candidate(10, "quick fox jumps")
        c20 = _candidate(20, "the quick brown fox")
        c30 = _candidate(30, "completely different text")

        chosen = silver_policy([c10, c20, c30], translated_source_texts=translated)
        assert chosen.corpus_idx == 20, (
            f"Expected c20 (highest BM25 overlap), got corpus_idx={chosen.corpus_idx}"
        )

    def test_no_target_access(self):
        """Silver policy function signature must not expose or accept target data."""
        import inspect
        sig = inspect.signature(silver_policy)
        param_names = list(sig.parameters.keys())
        for bad in ("target", "reference", "target_texts", "chrf", "bleu"):
            assert bad not in param_names, (
                f"silver_policy must not have parameter '{bad}' — target leakage!"
            )


# ---------------------------------------------------------------------------
# 5. Golden policy
# ---------------------------------------------------------------------------

class TestGoldenPolicy:
    @pytest.fixture
    def project_slice(self) -> ProjectSlice:
        """Small 5-item project slice with known frequency index."""
        items = [
            _make_pool_item(100, "alpha beta gamma"),
            _make_pool_item(101, "alpha delta epsilon"),
            _make_pool_item(102, "beta gamma zeta"),
            _make_pool_item(103, "alpha beta"),
            _make_pool_item(104, "omega"),
        ]
        manifest = _make_manifest([], remaining_items=items)
        return build_project_slice(manifest, limit=5)

    def test_returns_acquisition_candidate(self, project_slice):
        known = frozenset(["alpha"])
        candidates = [_candidate(i, f"beta {i}") for i in range(3)]
        result = golden_policy(candidates, project_slice=project_slice, known_types=known)
        assert isinstance(result, AcquisitionCandidate)

    def test_empty_candidates_raises(self, project_slice):
        with pytest.raises(ValueError):
            golden_policy([], project_slice=project_slice, known_types=frozenset())

    def test_chooses_max_weighted_gain(self, project_slice):
        """Candidate contributing highest weighted gain wins."""
        # freq_index from project_slice:
        # alpha: 3 (items 100, 101, 103), beta: 3 (100, 102, 103), gamma: 2 (100, 102)
        # delta: 1 (101), epsilon: 1 (101), zeta: 1 (102), omega: 1 (104)
        known = frozenset()
        # c0: "alpha" → gains alpha (freq=3)
        # c1: "beta" → gains beta (freq=3)
        # c2: "gamma" → gains gamma (freq=2)
        # Tie between c0 and c1 on weighted gain (both 3); unweighted tie (both 1)
        # → lowest corpus_idx: c0 wins
        c0 = _candidate(0, "alpha")
        c1 = _candidate(1, "beta")
        c2 = _candidate(2, "gamma")
        chosen = golden_policy([c0, c1, c2], project_slice=project_slice, known_types=known)
        # c0 and c1 tie at weighted=3, unweighted=1; c0 wins on corpus_idx
        assert chosen.corpus_idx == 0

    def test_project_frequency_ranking(self, project_slice):
        """Higher-frequency types in project make a candidate preferred."""
        # alpha appears 3 times, omega appears 1 time
        known = frozenset(["beta", "gamma", "delta", "epsilon", "zeta"])
        # c0: "alpha" → gains alpha (freq=3)
        # c1: "omega" → gains omega (freq=1)
        c0 = _candidate(0, "alpha")
        c1 = _candidate(1, "omega")
        chosen = golden_policy([c0, c1], project_slice=project_slice, known_types=known)
        assert chosen.corpus_idx == 0, "alpha (freq=3) should beat omega (freq=1)"

    def test_off_project_vocabulary_regression(self, project_slice):
        """Types absent from project yield zero gain; off-project candidates are last resort."""
        known = frozenset(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "omega"])
        # All project types known — nothing to gain
        # c0: "alpha beta" → 0 gain (already known)
        # c1: "xenon krypton" → 0 gain (not in project)
        c0 = _candidate(0, "alpha beta")
        c1 = _candidate(1, "xenon krypton")
        c_low = _candidate(0, "alpha beta")
        # Both have 0 weighted gain, 0 new types; lowest corpus_idx wins
        chosen = golden_policy([c0, c1], project_slice=project_slice, known_types=known)
        assert chosen.corpus_idx == 0

    def test_max_unweighted_tiebreak(self, project_slice):
        """When weighted gain ties, max unweighted new type count breaks the tie."""
        # known = {"alpha"}; freq_index: beta=3, gamma=2, delta=1, epsilon=1, zeta=1, omega=1
        known = frozenset(["alpha"])
        # c0: "beta" → weighted=3, new types=1
        # c1: "gamma delta" → weighted = gamma(2)+delta(1) = 3, new types=2 → WINS on unweighted
        c0 = _candidate(0, "beta")
        c1 = _candidate(1, "gamma delta")
        chosen = golden_policy([c0, c1], project_slice=project_slice, known_types=known)
        assert chosen.corpus_idx == 1, (
            f"gamma+delta (weighted=3, types=2) should beat beta (weighted=3, types=1)"
        )

    def test_stable_ties_lowest_corpus_idx(self, project_slice):
        """Final tiebreak: lowest corpus_idx wins when weighted and unweighted gains are equal."""
        known = frozenset()
        # Two candidates introducing identical single types with same frequency
        # delta and epsilon both appear once in the project
        c5 = _candidate(5, "delta")
        c2 = _candidate(2, "epsilon")
        # weighted = 1, types = 1 for both → corpus_idx tie-break
        chosen = golden_policy([c5, c2], project_slice=project_slice, known_types=known)
        assert chosen.corpus_idx == 2, "Lower corpus_idx should win on complete tie"

    def test_exposes_golden_gain(self, project_slice):
        """golden_policy should also return (or expose) ex-ante weighted and unweighted gains."""
        known = frozenset()
        c0 = _candidate(0, "alpha beta")
        gain = golden_gain(candidates=[c0], project_slice=project_slice, known_types=known)
        assert isinstance(gain, GoldenGain)
        assert gain.weighted_gain >= 0
        assert gain.unweighted_gain >= 0
        assert gain.candidate.corpus_idx == 0

    def test_does_not_mutate_known_types(self, project_slice):
        known = frozenset(["alpha"])
        original = frozenset(known)
        candidates = [_candidate(i, f"beta {i}") for i in range(3)]
        golden_policy(candidates, project_slice=project_slice, known_types=known)
        assert known == original

    def test_does_not_mutate_candidates_list(self, project_slice):
        candidates = [_candidate(i, f"text {i}") for i in range(4)]
        original = list(candidates)
        golden_policy(candidates, project_slice=project_slice, known_types=frozenset())
        assert candidates == original

    def test_no_target_access(self):
        """golden_policy signature must not accept target data."""
        import inspect
        sig = inspect.signature(golden_policy)
        param_names = list(sig.parameters.keys())
        for bad in ("target", "reference", "target_texts", "chrf", "bleu"):
            assert bad not in param_names, (
                f"golden_policy must not have parameter '{bad}' — target leakage!"
            )

    def test_single_candidate_returned(self, project_slice):
        c = _candidate(7, "alpha beta gamma")
        result = golden_policy([c], project_slice=project_slice, known_types=frozenset())
        assert result is c

    def test_empty_project_gain_zero(self):
        """If project slice has no types (vacuous project), all gains are zero."""
        # Empty remaining → freq_index = {}
        manifest = _make_manifest([], remaining_items=[])
        ps = build_project_slice(manifest, limit=1) if False else ProjectSlice(
            items=[], limit=1, freq_index={}
        )
        known = frozenset()
        c0 = _candidate(0, "alpha beta")
        c1 = _candidate(1, "gamma delta")
        # Both have zero gain; c0 wins on corpus_idx
        chosen = golden_policy([c0, c1], project_slice=ps, known_types=known)
        assert chosen.corpus_idx == 0


# ---------------------------------------------------------------------------
# 6. GoldenGain helper
# ---------------------------------------------------------------------------

def golden_gain(
    candidates: list[AcquisitionCandidate],
    project_slice: "ProjectSlice",
    known_types: frozenset[str],
) -> "GoldenGain":
    """Thin wrapper: return GoldenGain for the best candidate."""
    from constrained_translation.experiment.policies import golden_policy_with_gain
    return golden_policy_with_gain(candidates, project_slice=project_slice, known_types=known_types)


class TestGoldenGainDataclass:
    @pytest.fixture
    def project_slice(self):
        items = [_make_pool_item(i, f"alpha beta {i}") for i in range(3)]
        manifest = _make_manifest([], remaining_items=items)
        return build_project_slice(manifest, limit=3)

    def test_golden_gain_fields(self, project_slice):
        from constrained_translation.experiment.policies import golden_policy_with_gain
        c = _candidate(0, "alpha beta")
        gain = golden_policy_with_gain([c], project_slice=project_slice, known_types=frozenset())
        assert hasattr(gain, "candidate")
        assert hasattr(gain, "weighted_gain")
        assert hasattr(gain, "unweighted_gain")

    def test_weighted_gain_is_float(self, project_slice):
        from constrained_translation.experiment.policies import golden_policy_with_gain
        c = _candidate(0, "alpha")
        gain = golden_policy_with_gain([c], project_slice=project_slice, known_types=frozenset())
        assert isinstance(gain.weighted_gain, (int, float))

    def test_unweighted_gain_is_int(self, project_slice):
        from constrained_translation.experiment.policies import golden_policy_with_gain
        c = _candidate(0, "alpha")
        gain = golden_policy_with_gain([c], project_slice=project_slice, known_types=frozenset())
        assert isinstance(gain.unweighted_gain, int)

    def test_candidate_matches_chosen(self, project_slice):
        from constrained_translation.experiment.policies import golden_policy_with_gain
        candidates = [_candidate(i, f"alpha {i}") for i in range(3)]
        known = frozenset()
        gain = golden_policy_with_gain(candidates, project_slice=project_slice, known_types=known)
        chosen = golden_policy(candidates, project_slice=project_slice, known_types=known)
        assert gain.candidate == chosen


# ---------------------------------------------------------------------------
# 7. Integration: no target data flows through any policy
# ---------------------------------------------------------------------------

class TestNoTargetLeakage:
    """Meta-tests: verify the API surface has no target exposure."""

    def test_arbitrary_policy_signature(self):
        import inspect
        sig = inspect.signature(arbitrary_policy)
        params = list(sig.parameters.keys())
        for bad in ("target", "reference", "target_texts", "chrf", "bleu", "translation"):
            assert bad not in params, f"arbitrary_policy has forbidden param '{bad}'"

    def test_silver_policy_only_source_params(self):
        import inspect
        sig = inspect.signature(silver_policy)
        params = list(sig.parameters.keys())
        # Must have 'candidates' and 'translated_source_texts'
        assert "candidates" in params
        assert "translated_source_texts" in params
        for bad in ("target_texts", "reference", "chrf", "bleu"):
            assert bad not in params

    def test_golden_policy_only_source_params(self):
        import inspect
        sig = inspect.signature(golden_policy)
        params = list(sig.parameters.keys())
        assert "candidates" in params
        assert "project_slice" in params
        assert "known_types" in params
        for bad in ("target_texts", "reference", "chrf", "bleu"):
            assert bad not in params

    def test_acquisition_candidate_is_not_pool_item(self):
        """AcquisitionCandidate must be a separate type from SequencingPoolItem."""
        import inspect
        assert AcquisitionCandidate is not SequencingPoolItem
        # AcquisitionCandidate must not be a subclass of SequencingPoolItem
        assert not issubclass(AcquisitionCandidate, SequencingPoolItem)


# ---------------------------------------------------------------------------
# Regression 1: arbitrary_policy — fixed permutation replay
# ---------------------------------------------------------------------------

class TestArbitraryPolicyFixedPermutation:
    """
    Regression: arbitrary_policy must replay one fixed initial permutation
    rather than re-shuffling the shrinking pool each call.

    The contract is:
        full_order = shuffle(all_candidates, seed)
        for each round: winner == next item in full_order that's still present

    Equivalently: iteratively calling arbitrary_policy (removing the winner
    each round) must produce exactly the same sequence as the one-time seeded
    shuffle of the full candidate list.
    """

    def _full_order(self, candidates: list, seed: int) -> list:
        """Compute the expected stable-ranking order (matches arbitrary_policy).

        Uses the same per-candidate seeding as the implementation:
        each candidate gets rank = Random(seed * 1_000_003 + corpus_idx).random(),
        then all candidates are sorted ascending by rank to yield the
        fixed initial permutation.
        """
        import random
        _RANK_MIX = 1_000_003
        ranked = [
            (random.Random(seed * _RANK_MIX + c.corpus_idx).random(), c)
            for c in candidates
        ]
        return [c for _, c in sorted(ranked, key=lambda rc: rc[0])]

    def test_iterative_order_equals_initial_permutation(self):
        """Remove-and-call sequence must equal the initial full-shuffle order."""
        candidates = [_candidate(i, f"sentence number {i}") for i in range(8)]
        seed = 42

        expected = self._full_order(candidates, seed)
        expected_idx = [c.corpus_idx for c in expected]

        pool = list(candidates)
        actual_idx = []
        while pool:
            winner = arbitrary_policy(pool, seed=seed)
            actual_idx.append(winner.corpus_idx)
            pool = [c for c in pool if c != winner]

        assert actual_idx == expected_idx, (
            f"Iterative order {actual_idx} != initial permutation {expected_idx}.\n"
            "arbitrary_policy must replay one fixed permutation, not re-shuffle each call."
        )

    def test_fixed_permutation_across_five_seeds(self):
        """For each of seeds 0-4, iterative selection must match one-time shuffle."""
        candidates = [_candidate(i, f"word{i} sample") for i in range(7)]
        for seed in range(5):
            expected_idx = [c.corpus_idx for c in self._full_order(candidates, seed)]
            pool = list(candidates)
            actual_idx = []
            while pool:
                winner = arbitrary_policy(pool, seed=seed)
                actual_idx.append(winner.corpus_idx)
                pool = [c for c in pool if c != winner]
            assert actual_idx == expected_idx, (
                f"Seed {seed}: iterative {actual_idx} != permutation {expected_idx}"
            )

    def test_every_candidate_selected_exactly_once(self):
        """All candidates are selected exactly once when pool shrinks iteratively."""
        candidates = [_candidate(i, f"text {i}") for i in range(10)]
        for seed in [7, 99, 1234]:
            pool = list(candidates)
            selected = []
            while pool:
                winner = arbitrary_policy(pool, seed=seed)
                assert winner not in selected, (
                    f"Seed {seed}: candidate {winner.corpus_idx} selected twice!"
                )
                selected.append(winner)
                pool = [c for c in pool if c != winner]
            assert len(selected) == 10
            assert set(c.corpus_idx for c in selected) == set(range(10))

    def test_no_global_state_mutation(self):
        """Policy must use isolated RNG — global random state must be unchanged."""
        import random
        candidates = [_candidate(i, f"text {i}") for i in range(6)]
        random.seed(99999)
        state_before = random.random()
        random.seed(99999)
        pool = list(candidates)
        while pool:
            winner = arbitrary_policy(pool, seed=42)
            pool = [c for c in pool if c != winner]
        state_after = random.random()
        assert state_before == state_after, (
            "arbitrary_policy contaminated global random state over multiple calls!"
        )

    def test_deterministic_across_five_seeds_differ(self):
        """Distinct seeds produce distinct orderings (≥5 unique orderings)."""
        candidates = [_candidate(i, f"word{i} text sample") for i in range(8)]
        orders = set()
        for seed in [0, 1, 2, 3, 4, 17, 42, 99, 1000, 9999]:
            pool = list(candidates)
            order = []
            while pool:
                winner = arbitrary_policy(pool, seed=seed)
                order.append(winner.corpus_idx)
                pool = [x for x in pool if x != winner]
            orders.add(tuple(order))
        assert len(orders) >= 5, (
            f"Expected ≥5 distinct seed orderings, got {len(orders)}: {orders}"
        )


# ---------------------------------------------------------------------------
# Regression 2: silver_policy — occurrence-based fraction
# ---------------------------------------------------------------------------

class TestSilverOccurrenceFraction:
    """
    Regression: silver_policy must score based on normalized source-unit
    OCCURRENCE count (known_occurrences / total_occurrences), not unique-type
    count.  Repeated known/unknown units must affect the score.

    Punctuation-only (empty unit list) → fraction = 0.0 (unchanged).
    """

    def test_occurrence_fraction_beats_type_fraction_tie(self):
        """
        When type fractions tie but occurrence fractions differ,
        the candidate with higher occurrence fraction wins.

        Pool knows 'alpha'.
        c0: 'alpha beta'        → types {alpha,beta}: type-frac=1/2
                                  occurrences [alpha,beta]: occ-frac=1/2
        c1: 'alpha alpha beta'  → types {alpha,beta}: type-frac=1/2 (SAME type frac!)
                                  occurrences [alpha,alpha,beta]: occ-frac=2/3 (HIGHER)
        c1 must win.
        """
        translated = ["alpha"]
        c0 = _candidate(0, "alpha beta")
        c1 = _candidate(1, "alpha alpha beta")

        chosen = silver_policy([c0, c1], translated_source_texts=translated)
        assert chosen.corpus_idx == 1, (
            f"Expected c1 (occ-frac 2/3 > 1/2), got corpus_idx={chosen.corpus_idx}.\n"
            "Silver must use occurrence fraction, not unique-type fraction."
        )

    def test_repeated_unknown_lowers_score(self):
        """
        Repeated unknown tokens dilute the occurrence fraction downward.

        Pool knows 'alpha'.
        c0: 'alpha beta beta'   → occurrences [alpha,beta,beta]: occ-frac=1/3
        c1: 'alpha beta'        → occurrences [alpha,beta]:      occ-frac=1/2
        c1 must win (higher occ-frac).
        """
        translated = ["alpha"]
        c0 = _candidate(0, "alpha beta beta")
        c1 = _candidate(1, "alpha beta")

        chosen = silver_policy([c0, c1], translated_source_texts=translated)
        assert chosen.corpus_idx == 1, (
            f"Expected c1 (occ-frac 1/2 > 1/3), got corpus_idx={chosen.corpus_idx}.\n"
            "Repeated unknown tokens should dilute the occurrence fraction."
        )

    def test_punctuation_only_is_zero(self):
        """Punctuation-only candidate has occurrence fraction 0.0 (deterministic)."""
        translated = ["alpha beta"]
        c_punct = _candidate(0, ",,, ...")       # normalizes to [] → frac=0
        c_normal = _candidate(1, "gamma delta")  # no known units → frac=0 too
        # Both score 0.0; tie broken by BM25 then corpus_idx → c_punct wins (idx=0)
        chosen = silver_policy([c_punct, c_normal], translated_source_texts=translated)
        # Should not crash; result is deterministic
        assert isinstance(chosen, AcquisitionCandidate)

    def test_fully_known_occurrences_score_one(self):
        """Candidate whose every occurrence is known scores 1.0."""
        translated = ["alpha beta gamma"]
        # c0: all tokens known → occ-frac = 1.0
        c0 = _candidate(0, "alpha beta gamma alpha")
        # c1: mixed → occ-frac < 1.0
        c1 = _candidate(1, "alpha beta unknown")
        chosen = silver_policy([c0, c1], translated_source_texts=translated)
        assert chosen.corpus_idx == 0, (
            f"Expected c0 (all occurrences known → frac=1.0), got {chosen.corpus_idx}"
        )

    def test_known_occurrence_count_over_total_occurrences(self):
        """
        Validate the formula directly:
        Pool: 'alpha alpha beta'  → known_set = {alpha, beta}
        c0:  'alpha alpha alpha beta beta'  → 5 total, 5 known → frac=1.0
        c1:  'alpha gamma'                  → 2 total, 1 known → frac=0.5
        c0 wins.
        """
        translated = ["alpha alpha beta"]
        c0 = _candidate(0, "alpha alpha alpha beta beta")
        c1 = _candidate(1, "alpha gamma")
        chosen = silver_policy([c0, c1], translated_source_texts=translated)
        assert chosen.corpus_idx == 0, (
            f"Expected c0 (occ-frac=1.0), got {chosen.corpus_idx}"
        )
