"""tests/constrained_translation/experiment/test_corpus_coverage.py

Strict TDD tests for corpus coverage accounting.
Written BEFORE the implementation (RED phase).

Required behaviour (per docs/plans/2026-07-10-slipstream-sequencing.md §Task 2):
1. Reuse normalize_source_units exactly; no second tokenizer.
2. Freeze operational project as first 200 items from manifest.remaining in corpus_idx order
   (or fewer only when explicitly configured); deterministic helper; fail for invalid limits.
3. Build occurrence-frequency index over the fixed project slice.
4. Given translated source cells: known types, unknown types, weighted unknown occurrences,
   weighted supported occurrences, support percentage, fully-supported project cells.
5. Return deterministic unsupported spans/units (source-side only, no target/reference input).
6. Detect false source abstention.
7. Compute frequency-weighted marginal gain and unweighted new-type gain.
8. Prove exact ex-ante gain equals ex-post weighted-unknown reduction after adding candidate.
9. Handle repeated units, empty project, punctuation/case normalization, Unicode/RTL, candidates
   with types absent from project.
10. Pure/deterministic; independent of GPU/backend/target text.
"""
from __future__ import annotations

import pytest

from constrained_translation.experiment.corpus_coverage import (
    ProjectSlice,
    build_project_slice,
    compute_coverage_stats,
    detect_false_source_abstentions,
    marginal_gain,
    CoverageStats,
)
from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(corpus_idx: int, source_text: str) -> SequencingPoolItem:
    """Helper: make a minimal SequencingPoolItem with only source_text."""
    return SequencingPoolItem(
        corpus_idx=corpus_idx,
        item_id=f"IDX:{corpus_idx}",
        source_text=source_text,
        target_texts={},
    )


def _make_manifest(remaining_items: list[SequencingPoolItem]) -> SequencingManifest:
    """Helper: make a minimal SequencingManifest with given remaining items."""
    return SequencingManifest(
        languages=[],
        random_seed=0,
        seed_size=0,
        acq_size=0,
        eval_size=0,
        seed=[],
        acquisition=[],
        fixed_eval=[],
        remaining=remaining_items,
        digest="",
    )


@pytest.fixture
def small_manifest():
    """A manifest with 10 remaining items covering a small vocabulary."""
    items = [
        _make_item(0,  "In the beginning God created the heavens"),
        _make_item(1,  "The earth was without form and void"),
        _make_item(2,  "God said let there be light"),
        _make_item(3,  "God saw the light was good"),
        _make_item(4,  "God called the light Day and darkness Night"),
        _make_item(5,  "God made the firmament above the waters"),
        _make_item(6,  "God blessed them saying be fruitful"),
        _make_item(7,  "God saw every thing that he had made"),
        _make_item(8,  "Thus the heavens and the earth were finished"),
        _make_item(9,  "God ended his work on the seventh day"),
    ]
    return _make_manifest(items)


@pytest.fixture
def manifest_250():
    """A manifest with 250 remaining items (corpus_idx 0..249) for default-limit testing."""
    items = [
        _make_item(i, f"word{i} alpha beta gamma delta epsilon zeta")
        for i in range(250)
    ]
    return _make_manifest(items)


@pytest.fixture
def manifest_150():
    """A manifest with only 150 remaining items for under-limit testing."""
    items = [
        _make_item(i, f"word{i} alpha beta gamma")
        for i in range(150)
    ]
    return _make_manifest(items)


# ---------------------------------------------------------------------------
# Class 1: ProjectSlice — data class
# ---------------------------------------------------------------------------

class TestProjectSlice:
    def test_has_items(self, small_manifest):
        ps = build_project_slice(small_manifest)
        assert hasattr(ps, "items")

    def test_items_are_sequencing_pool_items(self, small_manifest):
        ps = build_project_slice(small_manifest)
        for item in ps.items:
            assert isinstance(item, SequencingPoolItem)

    def test_has_limit(self, small_manifest):
        ps = build_project_slice(small_manifest)
        assert hasattr(ps, "limit")

    def test_has_freq_index(self, small_manifest):
        ps = build_project_slice(small_manifest)
        assert hasattr(ps, "freq_index")

    def test_freq_index_is_dict(self, small_manifest):
        ps = build_project_slice(small_manifest)
        assert isinstance(ps.freq_index, dict)


# ---------------------------------------------------------------------------
# Class 2: build_project_slice — first 200 in corpus_idx order
# ---------------------------------------------------------------------------

class TestBuildProjectSlice:
    def test_default_limit_200_items_available(self, manifest_250):
        """When ≥200 remaining items, slice is exactly 200."""
        ps = build_project_slice(manifest_250)
        assert len(ps.items) == 200

    def test_default_limit_fewer_than_200_items(self, manifest_150):
        """When <200 remaining items, slice is all of them."""
        ps = build_project_slice(manifest_150)
        assert len(ps.items) == 150

    def test_items_in_corpus_idx_order(self, manifest_250):
        """Items must be ordered by corpus_idx ascending."""
        ps = build_project_slice(manifest_250)
        idxs = [item.corpus_idx for item in ps.items]
        assert idxs == sorted(idxs)

    def test_items_in_corpus_idx_order_small(self, small_manifest):
        ps = build_project_slice(small_manifest)
        idxs = [item.corpus_idx for item in ps.items]
        assert idxs == sorted(idxs)

    def test_takes_first_200_by_corpus_idx(self, manifest_250):
        """Must take the 200 items with smallest corpus_idx values."""
        ps = build_project_slice(manifest_250)
        expected_indices = sorted([item.corpus_idx for item in manifest_250.remaining])[:200]
        actual_indices = [item.corpus_idx for item in ps.items]
        assert actual_indices == expected_indices

    def test_unsorted_remaining_still_ordered(self):
        """Even if manifest.remaining is not sorted, output is corpus_idx sorted."""
        items = [
            _make_item(5, "five alpha beta"),
            _make_item(1, "one alpha beta"),
            _make_item(3, "three alpha beta"),
        ]
        m = _make_manifest(items)
        ps = build_project_slice(m, limit=10)
        idxs = [item.corpus_idx for item in ps.items]
        assert idxs == [1, 3, 5]

    def test_custom_limit_respected(self, small_manifest):
        ps = build_project_slice(small_manifest, limit=5)
        assert len(ps.items) == 5

    def test_limit_zero_raises(self, small_manifest):
        with pytest.raises(ValueError):
            build_project_slice(small_manifest, limit=0)

    def test_limit_negative_raises(self, small_manifest):
        with pytest.raises(ValueError):
            build_project_slice(small_manifest, limit=-1)

    def test_empty_remaining_returns_empty_slice(self):
        m = _make_manifest([])
        ps = build_project_slice(m)
        assert len(ps.items) == 0

    def test_limit_stored_on_slice(self, small_manifest):
        ps = build_project_slice(small_manifest, limit=5)
        assert ps.limit == 5

    def test_default_limit_200_stored(self, manifest_250):
        ps = build_project_slice(manifest_250)
        assert ps.limit == 200


# ---------------------------------------------------------------------------
# Class 3: CorpusCoverageIndex — occurrence-frequency index
# ---------------------------------------------------------------------------

class TestCorpusCoverageIndex:
    def test_freq_index_counts_tokens(self, small_manifest):
        """freq_index maps normalized type → total occurrence count in project slice."""
        ps = build_project_slice(small_manifest)
        # "god" appears in items 0,2,3,4,5,6,7,9 → 8 times
        assert ps.freq_index.get("god", 0) >= 2

    def test_freq_index_uses_normalize_source_units(self, small_manifest):
        """freq_index must use normalize_source_units (not a different tokenizer)."""
        ps = build_project_slice(small_manifest)
        # Punctuation-stripped, lowercased keys expected
        for key in ps.freq_index:
            assert key == key.lower()
            assert key == key.strip()
            # Key should match what normalize_source_units produces
            re_normalized = normalize_source_units(key)
            assert re_normalized == [key], (
                f"freq_index key {key!r} is not a normalized unit"
            )

    def test_freq_index_repeated_units_in_one_sentence(self):
        """Repeated units within one sentence count multiply."""
        item = _make_item(0, "the cat and the dog and the bird")
        m = _make_manifest([item])
        ps = build_project_slice(m)
        # "the" appears 3 times, "and" appears 2 times
        assert ps.freq_index.get("the", 0) == 3
        assert ps.freq_index.get("and", 0) == 2

    def test_freq_index_case_insensitive(self):
        """Case variants must merge into one type."""
        item = _make_item(0, "God god GOD created")
        m = _make_manifest([item])
        ps = build_project_slice(m)
        # all three "God/god/GOD" normalize to "god"
        assert ps.freq_index.get("god", 0) == 3

    def test_freq_index_punctuation_stripped(self):
        """Punctuation-only tokens don't appear in freq_index; punctuation attached
        to words is stripped before counting."""
        item = _make_item(0, "In the beginning,")
        m = _make_manifest([item])
        ps = build_project_slice(m)
        # "beginning," normalizes to "beginning"
        assert "beginning" in ps.freq_index
        # No punctuation-only key
        assert "," not in ps.freq_index

    def test_freq_index_empty_project(self):
        """Empty project → empty freq_index."""
        ps = build_project_slice(_make_manifest([]))
        assert ps.freq_index == {}

    def test_freq_index_accumulates_across_items(self):
        """Counts accumulate across multiple items."""
        items = [
            _make_item(0, "alpha beta"),
            _make_item(1, "alpha gamma"),
            _make_item(2, "alpha beta delta"),
        ]
        m = _make_manifest(items)
        ps = build_project_slice(m)
        assert ps.freq_index["alpha"] == 3
        assert ps.freq_index["beta"] == 2
        assert ps.freq_index["gamma"] == 1
        assert ps.freq_index["delta"] == 1

    def test_freq_index_only_covers_slice_not_full_remaining(self, manifest_250):
        """When remaining has >200 items, freq_index is built only over the first 200."""
        ps = build_project_slice(manifest_250)
        # Items 200-249 have unique word types not in the first 200
        # word200..word249 should NOT appear
        for i in range(200, 250):
            assert f"word{i}" not in ps.freq_index
        # word0..word199 should appear
        for i in range(200):
            assert f"word{i}" in ps.freq_index


# ---------------------------------------------------------------------------
# Class 4: compute_coverage_stats
# ---------------------------------------------------------------------------

class TestComputeCoverageStats:
    """compute_coverage_stats(project_slice, translated_source_cells) → CoverageStats."""

    def _project_3(self):
        """Small 3-item project with known freq_index."""
        items = [
            _make_item(0, "alpha beta gamma"),      # 3 unique types
            _make_item(1, "alpha delta epsilon"),    # 2 new types (delta, epsilon)
            _make_item(2, "gamma zeta"),             # 1 new type (zeta)
        ]
        return build_project_slice(_make_manifest(items))

    def test_returns_coverage_stats(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert isinstance(stats, CoverageStats)

    def test_known_types_empty_translated(self):
        """With no translated cells, known_types is empty."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert stats.known_types == frozenset()

    def test_unknown_types_all_project_types_when_no_translated(self):
        """With no translated cells, all project types are unknown."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        expected = frozenset(ps.freq_index.keys())
        assert stats.unknown_types == expected

    def test_known_types_after_translating_one_cell(self):
        """After translating a cell, its normalized units become known."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta gamma"])
        assert "alpha" in stats.known_types
        assert "beta" in stats.known_types
        assert "gamma" in stats.known_types

    def test_unknown_types_reduced_after_translation(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta gamma"])
        # alpha, beta, gamma are now known; delta, epsilon, zeta still unknown
        assert "alpha" not in stats.unknown_types
        assert "delta" in stats.unknown_types

    def test_weighted_unknown_occurrences_empty(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta gamma delta epsilon zeta"])
        assert stats.weighted_unknown_occurrences == 0

    def test_weighted_unknown_occurrences_counts_freq(self):
        """Weighted unknown = sum of freq_index[t] for t in unknown_types."""
        ps = self._project_3()
        # Only translate alpha, beta → delta, epsilon, zeta, gamma are unknown
        # freq: alpha=2, beta=1, gamma=2, delta=1, epsilon=1, zeta=1
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        # unknown: gamma(2), delta(1), epsilon(1), zeta(1) → total 5
        assert stats.weighted_unknown_occurrences == 5

    def test_weighted_supported_occurrences(self):
        """Weighted supported = sum of freq_index[t] for t in known_types."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        # known: alpha(2), beta(1) → total 3
        assert stats.weighted_supported_occurrences == 3

    def test_support_percentage_zero_when_none_translated(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert stats.support_percentage == 0.0

    def test_support_percentage_100_when_all_translated(self):
        ps = self._project_3()
        all_words = "alpha beta gamma delta epsilon zeta"
        stats = compute_coverage_stats(ps, translated_source_cells=[all_words])
        assert stats.support_percentage == 100.0

    def test_support_percentage_proportional(self):
        """support_percentage = 100 * weighted_supported / (weighted_supported + weighted_unknown)."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        # supported=3, unknown=5, total=8
        expected = 100.0 * 3 / 8
        assert abs(stats.support_percentage - expected) < 1e-9

    def test_support_percentage_empty_project(self):
        """Empty project: support_percentage=100 (vacuously all covered)."""
        ps = build_project_slice(_make_manifest([]))
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert stats.support_percentage == 100.0

    def test_fully_supported_project_cells(self):
        """A project cell is fully-supported if ALL its normalized types are known."""
        ps = self._project_3()
        # alpha, beta, gamma known → item0 ("alpha beta gamma") fully supported
        # delta, epsilon unknown → item1 NOT fully supported
        # gamma known, zeta unknown → item2 NOT fully supported
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta gamma"])
        assert stats.fully_supported_count == 1

    def test_fully_supported_zero_initially(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert stats.fully_supported_count == 0

    def test_fully_supported_all_when_everything_covered(self):
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta gamma delta epsilon zeta"])
        assert stats.fully_supported_count == len(ps.items)

    def test_known_types_uses_normalize_source_units(self):
        """translated_source_cells are normalized via normalize_source_units before lookup."""
        ps = self._project_3()
        # Pass uppercase/punctuated version — should still normalize to known types
        stats = compute_coverage_stats(ps, translated_source_cells=["Alpha, Beta!"])
        assert "alpha" in stats.known_types
        assert "beta" in stats.known_types

    def test_no_target_input_required(self):
        """compute_coverage_stats must accept only source strings, no target/reference."""
        ps = self._project_3()
        # This must not require target text as input
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        assert isinstance(stats, CoverageStats)

    def test_translated_cells_outside_project_vocab_still_known(self):
        """Types from translated cells not in project freq_index still become known."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["unknown_word"])
        assert "unknown_word" in stats.known_types

    def test_repeated_units_in_one_project_item(self):
        """If a project item has repeated units, freq_index reflects all occurrences."""
        item = _make_item(0, "the the the cat")
        ps = build_project_slice(_make_manifest([item]))
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        # "the" should appear 3 times in freq_index
        assert ps.freq_index.get("the") == 3
        # All unknown initially
        assert stats.weighted_unknown_occurrences == 4  # the(3) + cat(1)

    def test_weighted_unknown_plus_supported_equals_total(self):
        """weighted_unknown + weighted_supported must equal sum of all project frequencies."""
        ps = self._project_3()
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        total = sum(ps.freq_index.values())
        assert stats.weighted_unknown_occurrences + stats.weighted_supported_occurrences == total


# ---------------------------------------------------------------------------
# Class 5: unsupported spans — compatible with UNKDetector semantics
# ---------------------------------------------------------------------------

class TestUnsupportedSpans:
    """compute_coverage_stats returns unsupported_spans per project cell,
    compatible with existing UNKDetector UNKSpan semantics."""

    def test_unsupported_spans_present_on_stats(self):
        items = [_make_item(0, "alpha beta gamma")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert hasattr(stats, "unsupported_spans")

    def test_unsupported_spans_is_dict(self):
        items = [_make_item(0, "alpha beta gamma")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert isinstance(stats.unsupported_spans, dict)

    def test_unsupported_spans_keyed_by_corpus_idx(self):
        items = [_make_item(0, "alpha beta"), _make_item(1, "gamma delta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=[])
        assert 0 in stats.unsupported_spans
        assert 1 in stats.unsupported_spans

    def test_unsupported_spans_empty_when_fully_supported(self):
        items = [_make_item(0, "alpha beta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        assert stats.unsupported_spans[0] == ()

    def test_unsupported_spans_source_side_only(self):
        """No target text is passed; spans are derived purely from source."""
        items = [_make_item(0, "alpha beta unknown_word")]
        ps = build_project_slice(_make_manifest(items))
        # Only "alpha" and "beta" are known
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        spans = stats.unsupported_spans[0]
        assert len(spans) > 0
        # All spans have surface attribute (UNKSpan-compatible)
        for span in spans:
            assert hasattr(span, "surface")
            assert hasattr(span, "start_char")
            assert hasattr(span, "end_char")

    def test_unsupported_spans_deterministic(self):
        """Same inputs always produce same spans."""
        items = [_make_item(0, "alpha unknown1 beta unknown2")]
        ps = build_project_slice(_make_manifest(items))
        s1 = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        s2 = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        assert s1.unsupported_spans == s2.unsupported_spans

    def test_unsupported_units_per_item(self):
        """stats.unsupported_units maps corpus_idx → frozenset of unknown normalized types."""
        items = [_make_item(0, "alpha beta delta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        assert hasattr(stats, "unsupported_units")
        # beta and delta are unknown
        assert "beta" in stats.unsupported_units[0]
        assert "delta" in stats.unsupported_units[0]
        assert "alpha" not in stats.unsupported_units[0]


# ---------------------------------------------------------------------------
# Class 6: detect_false_source_abstentions
# ---------------------------------------------------------------------------

class TestDetectFalseSourceAbstentions:
    """detect_false_source_abstentions(unsupported_units, translated_source_cells)
    → dict[corpus_idx, frozenset[str]] of false abstentions.

    A false abstention occurs when a unit is marked unknown but its normalized
    form IS canonically present in the translated source evidence.
    """

    def test_no_false_abstentions_when_correct(self):
        """If all unknown types are genuinely absent from translated pool, no false abstentions."""
        items = [_make_item(0, "alpha beta unknown_word")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        false_abs = detect_false_source_abstentions(stats, translated_source_cells=["alpha beta"])
        # unknown_word is genuinely not in translated pool → no false abstention
        assert false_abs.get(0, frozenset()) == frozenset()

    def test_detects_false_abstention(self):
        """If a unit is unknown but IS in translated pool (e.g. normalization defect),
        it's a false abstention."""
        items = [_make_item(0, "Alpha Beta gamma")]
        ps = build_project_slice(_make_manifest(items))
        # Suppose gamma appears unknown but "gamma" WAS in translated cells
        # Simulate by giving a translated pool missing gamma initially for stats,
        # then detect_false_source_abstentions with the actual pool containing gamma.
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        # unknown_units for item 0: gamma
        # Now detect with actual translated pool that contains "Gamma" (normalizes to gamma)
        false_abs = detect_false_source_abstentions(stats, translated_source_cells=["alpha beta Gamma"])
        assert "gamma" in false_abs.get(0, frozenset())

    def test_no_false_abstention_when_unit_genuinely_absent(self):
        items = [_make_item(0, "alpha truly_unknown")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        false_abs = detect_false_source_abstentions(stats, translated_source_cells=["alpha"])
        assert "truly_unknown" not in false_abs.get(0, frozenset())

    def test_false_abstentions_empty_dict_when_no_unknowns(self):
        items = [_make_item(0, "alpha beta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        false_abs = detect_false_source_abstentions(stats, translated_source_cells=["alpha beta"])
        assert false_abs.get(0, frozenset()) == frozenset()

    def test_returns_dict(self):
        items = [_make_item(0, "alpha beta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        false_abs = detect_false_source_abstentions(stats, translated_source_cells=["alpha"])
        assert isinstance(false_abs, dict)

    def test_source_only_no_target_input(self):
        """detect_false_source_abstentions must not require target text."""
        items = [_make_item(0, "alpha beta")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        # Call with only source cells, no target
        result = detect_false_source_abstentions(stats, translated_source_cells=["alpha beta"])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Class 7: marginal_gain
# ---------------------------------------------------------------------------

class TestMarginalGain:
    """marginal_gain(candidate_source_text, project_slice, known_types)
    → (weighted_gain: float, new_type_count: int)

    weighted_gain: sum of freq_index[t] for t in candidate's normalized types
                   that are NOT yet in known_types AND are in freq_index.
    new_type_count: count of types from candidate not in known_types
                    (regardless of whether in freq_index).
    """

    def _ps_3(self):
        items = [
            _make_item(0, "alpha beta gamma"),
            _make_item(1, "alpha delta epsilon"),
            _make_item(2, "gamma zeta"),
        ]
        return build_project_slice(_make_manifest(items))

    def test_returns_tuple(self):
        ps = self._ps_3()
        result = marginal_gain("alpha beta", ps, known_types=frozenset())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_weighted_gain_all_new(self):
        """All candidate types are new → weighted_gain = sum of their freqs."""
        ps = self._ps_3()
        # freq: alpha=2, beta=1, gamma=2, delta=1, epsilon=1, zeta=1
        wg, nt = marginal_gain("alpha beta", ps, known_types=frozenset())
        assert wg == 2 + 1  # alpha(2) + beta(1)
        assert nt == 2

    def test_weighted_gain_some_already_known(self):
        """Already-known types don't contribute to gain."""
        ps = self._ps_3()
        known = frozenset(["alpha"])
        wg, nt = marginal_gain("alpha beta", ps, known_types=known)
        assert wg == 1  # only beta(1) is new
        assert nt == 1

    def test_weighted_gain_all_known_is_zero(self):
        ps = self._ps_3()
        known = frozenset(["alpha", "beta"])
        wg, nt = marginal_gain("alpha beta", ps, known_types=known)
        assert wg == 0
        assert nt == 0

    def test_weighted_gain_candidate_not_in_project(self):
        """Types in candidate not in project freq_index don't contribute to weighted_gain,
        but do contribute to new_type_count."""
        ps = self._ps_3()
        wg, nt = marginal_gain("totally_foreign", ps, known_types=frozenset())
        assert wg == 0      # not in project
        assert nt == 1      # new type even if not in project

    def test_weighted_gain_uses_normalize_source_units(self):
        """Candidate is normalized before lookup."""
        ps = self._ps_3()
        # "Alpha!" normalizes to "alpha"
        wg1, nt1 = marginal_gain("Alpha!", ps, known_types=frozenset())
        wg2, nt2 = marginal_gain("alpha", ps, known_types=frozenset())
        assert wg1 == wg2
        assert nt1 == nt2

    def test_weighted_gain_empty_candidate(self):
        ps = self._ps_3()
        wg, nt = marginal_gain("", ps, known_types=frozenset())
        assert wg == 0
        assert nt == 0

    def test_weighted_gain_punctuation_only_candidate(self):
        ps = self._ps_3()
        wg, nt = marginal_gain(",,, --- ???", ps, known_types=frozenset())
        assert wg == 0
        assert nt == 0

    def test_unicode_rtl_candidate(self):
        """Unicode/RTL strings must not raise and must process deterministically."""
        items = [_make_item(0, "شalom peace")]
        ps = build_project_slice(_make_manifest(items))
        wg, nt = marginal_gain("شalom peace", ps, known_types=frozenset())
        # Should not raise; result is deterministic
        wg2, nt2 = marginal_gain("شalom peace", ps, known_types=frozenset())
        assert wg == wg2
        assert nt == nt2


# ---------------------------------------------------------------------------
# Class 8: ex-ante gain == ex-post weighted-unknown reduction
# ---------------------------------------------------------------------------

class TestExAnteExPostCalibration:
    """Prove that the ex-ante weighted gain equals the ex-post reduction in
    weighted_unknown_occurrences after adding the candidate to translated pool."""

    def _ps_3(self):
        items = [
            _make_item(0, "alpha beta gamma"),
            _make_item(1, "alpha delta epsilon"),
            _make_item(2, "gamma zeta"),
        ]
        return build_project_slice(_make_manifest(items))

    def test_exante_equals_expost_reduction_from_empty(self):
        """Adding candidate to empty translated pool: ex-ante gain == ex-post reduction."""
        ps = self._ps_3()
        translated = []
        candidate = "alpha beta"
        stats_before = compute_coverage_stats(ps, translated_source_cells=translated)
        known_before = stats_before.known_types
        wg_exante, _ = marginal_gain(candidate, ps, known_types=known_before)

        translated_after = translated + [candidate]
        stats_after = compute_coverage_stats(ps, translated_source_cells=translated_after)
        reduction = stats_before.weighted_unknown_occurrences - stats_after.weighted_unknown_occurrences

        assert wg_exante == reduction

    def test_exante_equals_expost_reduction_from_partial(self):
        """Adding candidate to partially-translated pool: ex-ante == ex-post."""
        ps = self._ps_3()
        translated = ["alpha"]
        candidate = "beta gamma"
        stats_before = compute_coverage_stats(ps, translated_source_cells=translated)
        known_before = stats_before.known_types
        wg_exante, _ = marginal_gain(candidate, ps, known_types=known_before)

        translated_after = translated + [candidate]
        stats_after = compute_coverage_stats(ps, translated_source_cells=translated_after)
        reduction = stats_before.weighted_unknown_occurrences - stats_after.weighted_unknown_occurrences

        assert wg_exante == reduction

    def test_exante_equals_expost_all_types_covered(self):
        """When all types already covered, both ex-ante and ex-post reduction are zero."""
        ps = self._ps_3()
        all_words = "alpha beta gamma delta epsilon zeta"
        translated = [all_words]
        stats_before = compute_coverage_stats(ps, translated_source_cells=translated)
        wg_exante, _ = marginal_gain("alpha", ps, known_types=stats_before.known_types)
        translated_after = translated + ["alpha"]
        stats_after = compute_coverage_stats(ps, translated_source_cells=translated_after)
        reduction = stats_before.weighted_unknown_occurrences - stats_after.weighted_unknown_occurrences

        assert wg_exante == 0
        assert reduction == 0
        assert wg_exante == reduction

    def test_exante_equals_expost_repeated_types_in_project(self):
        """Repeated types handled correctly in both ex-ante and ex-post."""
        items = [
            _make_item(0, "the cat sat on the mat the mat"),  # the=3, mat=2, cat=1, sat=1, on=1
            _make_item(1, "the dog ran"),                      # the=1, dog=1, ran=1
        ]
        ps = build_project_slice(_make_manifest(items))
        translated = []
        candidate = "the cat"
        stats_before = compute_coverage_stats(ps, translated_source_cells=translated)
        wg_exante, _ = marginal_gain(candidate, ps, known_types=stats_before.known_types)

        translated_after = [candidate]
        stats_after = compute_coverage_stats(ps, translated_source_cells=translated_after)
        reduction = stats_before.weighted_unknown_occurrences - stats_after.weighted_unknown_occurrences

        assert wg_exante == reduction


# ---------------------------------------------------------------------------
# Class 9: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_project_coverage_stats(self):
        ps = build_project_slice(_make_manifest([]))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha beta"])
        assert stats.fully_supported_count == 0
        assert stats.support_percentage == 100.0
        assert stats.weighted_unknown_occurrences == 0
        assert stats.weighted_supported_occurrences == 0

    def test_project_item_with_only_punctuation(self):
        """Project item that normalizes to empty: no contribution to freq_index."""
        item = _make_item(0, ",,, --- ???")
        ps = build_project_slice(_make_manifest([item]))
        assert ps.freq_index == {}

    def test_unicode_rtl_source_in_project(self):
        """Unicode/RTL source strings are handled safely and deterministically."""
        items = [
            _make_item(0, "שָׁלוֹם עוֹלָם"),   # Hebrew
            _make_item(1, "مرحبا بالعالم"),    # Arabic
        ]
        ps = build_project_slice(_make_manifest(items))
        stats1 = compute_coverage_stats(ps, translated_source_cells=[])
        stats2 = compute_coverage_stats(ps, translated_source_cells=[])
        assert stats1.weighted_unknown_occurrences == stats2.weighted_unknown_occurrences

    def test_candidate_with_types_absent_from_project(self):
        """marginal_gain with candidate whose types aren't in project → weighted_gain=0."""
        items = [_make_item(0, "alpha beta")]
        ps = build_project_slice(_make_manifest(items))
        wg, nt = marginal_gain("xyz xyz_foreign", ps, known_types=frozenset())
        assert wg == 0
        assert nt == 2  # new types even though not in project

    def test_repeated_units_within_sentence_occurrence_count(self):
        """freq_index correctly counts all occurrences within a sentence."""
        item = _make_item(0, "foo foo foo bar")
        ps = build_project_slice(_make_manifest([item]))
        assert ps.freq_index["foo"] == 3
        assert ps.freq_index["bar"] == 1

    def test_support_percentage_float(self):
        """support_percentage is a float in [0, 100]."""
        items = [_make_item(0, "alpha beta gamma")]
        ps = build_project_slice(_make_manifest(items))
        stats = compute_coverage_stats(ps, translated_source_cells=["alpha"])
        assert isinstance(stats.support_percentage, float)
        assert 0.0 <= stats.support_percentage <= 100.0

    def test_deterministic_repeated_calls(self):
        """All functions return identical output on repeated calls."""
        items = [_make_item(i, f"word{i} alpha beta") for i in range(5)]
        ps1 = build_project_slice(_make_manifest(items))
        ps2 = build_project_slice(_make_manifest(items))
        assert ps1.freq_index == ps2.freq_index

        s1 = compute_coverage_stats(ps1, translated_source_cells=["alpha"])
        s2 = compute_coverage_stats(ps2, translated_source_cells=["alpha"])
        assert s1.known_types == s2.known_types
        assert s1.unknown_types == s2.unknown_types
        assert s1.weighted_unknown_occurrences == s2.weighted_unknown_occurrences

    def test_no_gpu_or_backend_dependency(self):
        """Module must be importable and functional with no GPU/backend."""
        # Simply importing and running pure Python proves this.
        from constrained_translation.experiment.corpus_coverage import build_project_slice
        item = _make_item(0, "hello world")
        ps = build_project_slice(_make_manifest([item]))
        assert "hello" in ps.freq_index
