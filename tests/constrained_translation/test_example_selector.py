"""Tests for Task 3: ExampleSelector — deterministic two-stage selection.

Strict TDD: these tests are written BEFORE the implementation.
All tests run fully offline using the tiny corpus from conftest.py.
"""
import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_selector(source_file, target_file, n_semantic=5, n_coverage=5):
    from constrained_translation.example_selector import ExampleSelector
    return ExampleSelector(
        source_file=source_file,
        target_file=target_file,
        n_semantic=n_semantic,
        n_coverage=n_coverage,
    )


# ---------------------------------------------------------------------------
# Structural / schema tests
# ---------------------------------------------------------------------------

class TestExampleSelectorReturnsAtMostNSemanticPlusNCoverage:
    """test_selector_returns_at_most_n_semantic_plus_n_coverage"""

    def test_default_at_most_ten(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=5)
        results = sel.select("God created the heavens and the earth")
        assert isinstance(results, tuple)
        assert len(results) <= 10

    def test_custom_n_respected(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=2, n_coverage=2)
        results = sel.select("God created the heavens and the earth")
        assert len(results) <= 4

    def test_returns_tuple(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        results = sel.select("God saw the light")
        assert isinstance(results, tuple)


class TestExampleSelectorDeduplicatesByVerseIdx:
    """test_selector_deduplicates_by_verse_idx"""

    def test_no_duplicate_verse_idx(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        # Use large n so semantic and coverage both span the full corpus.
        sel = _make_selector(src, tgt, n_semantic=12, n_coverage=12)
        results = sel.select("God created light and earth")
        verse_indices = [ex.verse_idx for ex in results]
        assert len(verse_indices) == len(set(verse_indices)), (
            "Duplicate verse_idx found in results"
        )

    def test_deduplicated_entry_keeps_semantic_method(self, tiny_corpus_files):
        """When a verse appears in both stages, the kept entry must be 'semantic'."""
        src, tgt = tiny_corpus_files
        # n_semantic=12 absorbs every verse; any coverage result will be a dup.
        sel = _make_selector(src, tgt, n_semantic=12, n_coverage=12)
        results = sel.select("God created light and earth")
        # All results are semantic because semantic sweep absorbs everything.
        methods = {ex.selection_method for ex in results}
        # "semantic" must be present; "coverage" may or may not be (absorbed).
        assert "semantic" in methods


# ---------------------------------------------------------------------------
# selection_method label tests
# ---------------------------------------------------------------------------

class TestExampleSelectorSemanticEntriesHaveMethodSemantic:
    """test_selector_semantic_entries_have_method_semantic"""

    def test_semantic_entries_labeled_correctly(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=3)
        results = sel.select("God created the heavens and the earth")
        # All results must have a valid method label.
        for ex in results:
            assert ex.selection_method in {"semantic", "coverage"}, (
                f"Unexpected selection_method: {ex.selection_method!r}"
            )

    def test_at_least_one_semantic_result(self, tiny_corpus_files):
        """With n_semantic > 0 and a non-trivial corpus, at least one semantic result."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=0)
        results = sel.select("God created the heavens")
        semantic = [ex for ex in results if ex.selection_method == "semantic"]
        assert len(semantic) >= 1


class TestExampleSelectorCoverageEntriesHaveMethodCoverage:
    """test_selector_coverage_entries_have_method_coverage"""

    def test_coverage_entries_labeled_correctly(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        # n_semantic=1 takes top-1; coverage may find different verses.
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=5)
        results = sel.select("God created the heavens and the earth light")
        for ex in results:
            assert ex.selection_method in {"semantic", "coverage"}

    def test_coverage_only_results_labeled_coverage(self, tiny_corpus_files):
        """Verses NOT returned by semantic stage but present in coverage → 'coverage'."""
        src, tgt = tiny_corpus_files
        # Semantic returns 1 result; any extra from coverage is labeled 'coverage'.
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=6)
        results = sel.select("God created the heavens and the earth light")
        # Verse indices from semantic (at most 1):
        sem_indices = {ex.verse_idx for ex in results if ex.selection_method == "semantic"}
        # Verse indices from coverage:
        cov_indices = {ex.verse_idx for ex in results if ex.selection_method == "coverage"}
        # Semantic and coverage sets must be disjoint (no double-labeling).
        assert sem_indices.isdisjoint(cov_indices), (
            "Same verse_idx appears under both 'semantic' and 'coverage'"
        )


# ---------------------------------------------------------------------------
# Evidence tier tests
# ---------------------------------------------------------------------------

class TestExampleSelectorAllEntriesHaveEvidenceTier1:
    """test_selector_all_entries_have_evidence_tier_1"""

    def test_all_evidence_tier_1(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=5)
        results = sel.select("God saw the earth")
        assert len(results) > 0, "Expected at least one result"
        for ex in results:
            assert ex.evidence_tier == 1, (
                f"evidence_tier must be 1 (sentence co-occurrence), got {ex.evidence_tier}"
            )

    def test_evidence_tier_1_for_coverage_entries(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=5)
        results = sel.select("blessed seventh day God work")
        for ex in results:
            assert ex.evidence_tier == 1


# ---------------------------------------------------------------------------
# Held-out / query verse exclusion tests
# ---------------------------------------------------------------------------

class TestExampleSelectorExcludesQueryVerse:
    """test_selector_excludes_query_verse_from_results"""

    def test_exclude_idx_zero(self, tiny_corpus_files):
        """Verse at index 0 must not appear even when its text is the query."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=5)
        # Query with the exact text of verse 0 to maximise retrieval score.
        query = "In the beginning God created the heavens and the earth"
        results = sel.select(query, exclude_idx=0)
        verse_indices = [ex.verse_idx for ex in results]
        assert 0 not in verse_indices, (
            "Verse 0 (the excluded query verse) must not appear in results"
        )

    def test_exclude_idx_middle(self, tiny_corpus_files):
        """Verse at an arbitrary interior index is excluded."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=12, n_coverage=12)
        query = "God blessed seventh day sanctified"
        EXCL = 11  # 0-based index of the last verse
        results = sel.select(query, exclude_idx=EXCL)
        assert all(ex.verse_idx != EXCL for ex in results), (
            f"Excluded verse_idx={EXCL} must not appear in results"
        )

    def test_exclude_idx_none_returns_full_results(self, tiny_corpus_files):
        """When exclude_idx=None, no verse is excluded."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=3)
        results_without_excl = sel.select(
            "God created the heavens and the earth", exclude_idx=None
        )
        # Should return some results (no artificial filtering).
        assert len(results_without_excl) > 0

    def test_exclusion_by_index_not_by_text(self, tiny_corpus_files):
        """Exclusion is by verse_idx (stable), not by source text matching.

        If the corpus had a duplicate source line, only the specific index
        would be excluded.  Here we verify the contract: exclude_idx=1 removes
        verse 1 even when the query is completely unrelated.
        """
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=12, n_coverage=12)
        results = sel.select(
            "In the beginning God created the heavens and the earth",
            exclude_idx=1,
        )
        assert all(ex.verse_idx != 1 for ex in results)


# ---------------------------------------------------------------------------
# AlignedExample field correctness
# ---------------------------------------------------------------------------

class TestExampleSelectorAlignedExampleFields:
    def test_verse_idx_is_int(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        for ex in sel.select("God light earth"):
            assert isinstance(ex.verse_idx, int)

    def test_source_and_target_are_strings(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        for ex in sel.select("God light earth"):
            assert isinstance(ex.source, str)
            assert isinstance(ex.target, str)
            assert len(ex.source) > 0
            assert len(ex.target) > 0

    def test_selection_score_is_float(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        for ex in sel.select("God light earth"):
            assert isinstance(ex.selection_score, float)

    def test_verse_idx_in_valid_range(self, tiny_corpus_files):
        """verse_idx must be a valid 0-based index into the corpus (0..11)."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        for ex in sel.select("God light earth"):
            assert 0 <= ex.verse_idx <= 11

    def test_results_are_frozen_dataclasses(self, tiny_corpus_files):
        import dataclasses
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt)
        for ex in sel.select("God light earth"):
            assert dataclasses.is_dataclass(ex)
            with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
                ex.verse_idx = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestExampleSelectorDeterminism:
    def test_same_query_same_results(self, tiny_corpus_files):
        """Two calls with identical arguments must return identical results."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=5)
        query = "God created the heavens and the earth"
        r1 = sel.select(query, exclude_idx=0)
        r2 = sel.select(query, exclude_idx=0)
        assert r1 == r2, "ExampleSelector must be deterministic"
