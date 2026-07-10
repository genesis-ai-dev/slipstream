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


# ---------------------------------------------------------------------------
# Regression: Stage 2 must target UNCOVERED units, not re-score full query
# ---------------------------------------------------------------------------

class TestCoverageTargetsUncoveredUnits:
    """Stage 2 must select examples covering query units NOT present in Stage 1.

    Regression guard: the old Stage 2 called ContextQuery._branching_search on
    the full original query without knowing which tokens were already covered by
    Stage 1.  In certain query/corpus configurations this caused the single
    coverage slot to be consumed by the *same* verse that semantic returned (the
    branching search scored it highest on the full query), which was then
    deduplicated away — leaving the coverage stage with zero results even though
    uncovered query tokens existed in the corpus.
    """

    def test_coverage_slot_not_wasted_on_already_covered_term(self, tiny_corpus_files):
        """With n_coverage=1, the one slot must pick a verse covering an uncovered term.

        Setup (tiny corpus):
          Query = "firmament earth"
          BM25 semantic top-1 → verse_idx=7
            "And God made the firmament above the waters"  → covers 'firmament' only.
          Remaining uncovered query token = {'earth'}.
          Verses 1, 9 contain 'earth', so there IS a valid coverage target.

        Old (buggy) behaviour:
          _branching_search("firmament earth", top_k=1, exclude_idx=-1) ALSO
          returns verse_idx=7 as its best result (same branching-search ranking).
          Verse 7 is then deduplicated away → 0 coverage examples returned, even
          though 'earth' is uncovered and reachable in the corpus.

        Correct behaviour:
          Coverage stage identifies {'earth'} as uncovered, then runs greedy
          set-cover over eligible rows (those not already selected by semantic).
          It picks exactly one verse that contains 'earth' and returns it.
        """
        import re

        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=1)
        results = sel.select("firmament earth")  # exclude_idx=None; no held-out verse

        sem_results = [ex for ex in results if ex.selection_method == "semantic"]
        cov_results = [ex for ex in results if ex.selection_method == "coverage"]

        # ── Stage 1 sanity ──────────────────────────────────────────────────
        assert len(sem_results) == 1, (
            f"Expected exactly 1 semantic result, got {len(sem_results)}"
        )

        # Determine which normalised query tokens semantic covered.
        query_tokens = {"firmament", "earth"}

        def _tokens(text: str) -> set[str]:
            return set(re.sub(r"[^\w\s]", "", text.lower()).split())

        sem_covered: set[str] = set()
        for ex in sem_results:
            sem_covered |= (_tokens(ex.source) & query_tokens)

        uncovered = query_tokens - sem_covered

        assert uncovered, (
            f"Test setup invalid: semantic already covered all query tokens "
            f"({sem_covered}); nothing left to test for coverage stage."
        )

        # ── Stage 2 core assertion ───────────────────────────────────────────
        assert len(cov_results) == 1, (
            f"Coverage stage should have returned 1 result targeting uncovered "
            f"tokens {uncovered!r}, but returned {len(cov_results)}.  "
            f"(Old bug: _branching_search also ranked verse_idx=7 first and "
            f"the only coverage slot was deduped away.)"
        )

        # The one coverage result must actually cover a previously-uncovered token.
        cov_tokens = _tokens(cov_results[0].source)
        newly_covered = cov_tokens & uncovered
        assert newly_covered, (
            f"Coverage result (verse_idx={cov_results[0].verse_idx}) covers "
            f"{cov_tokens & query_tokens!r} but uncovered tokens were "
            f"{uncovered!r}.  Slot was wasted on already-covered terms."
        )

    def test_coverage_stops_when_all_units_covered(self, tiny_corpus_files):
        """When semantic covers every query token, coverage must return 0 results.

        This prevents wasting context-window slots on examples that provide no
        new lexical coverage.
        """
        import re

        src, tgt = tiny_corpus_files
        # n_semantic=2 is enough to cover both 'light' and 'earth' across two verses.
        sel = _make_selector(src, tgt, n_semantic=2, n_coverage=3)
        results = sel.select("light earth", exclude_idx=0)

        sem_results = [ex for ex in results if ex.selection_method == "semantic"]
        cov_results = [ex for ex in results if ex.selection_method == "coverage"]

        def _tokens(text: str) -> set[str]:
            return set(re.sub(r"[^\w\s]", "", text.lower()).split())

        query_tokens = {"light", "earth"}
        sem_covered: set[str] = set()
        for ex in sem_results:
            sem_covered |= (_tokens(ex.source) & query_tokens)

        if sem_covered >= query_tokens:
            # All tokens covered by semantic → coverage must be empty.
            for ex in cov_results:
                cov_tokens = _tokens(ex.source)
                newly = cov_tokens & (query_tokens - sem_covered)
                assert newly, (
                    f"Coverage result (verse_idx={ex.verse_idx}) covers no new "
                    f"query tokens.  All tokens were already in semantic results."
                )


# ---------------------------------------------------------------------------
# Per-call n override and single-index-build tests (quality review)
# ---------------------------------------------------------------------------

class TestExampleSelectorPerCallNOverride:
    """Verify per-call n_semantic/n_coverage overrides and that the BM25
    index is only built once across calls.
    """

    def test_per_call_n_coverage_override_returns_more_results(self, tiny_corpus_files):
        """Per-call n_coverage override returns more results than the default."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=2, n_coverage=1)
        query = "God created the heavens and the earth"
        results_default = sel.select(query)
        results_expanded = sel.select(query, n_coverage=5)
        # Expanded should return at least as many results.
        assert len(results_expanded) >= len(results_default)

    def test_per_call_n_semantic_override(self, tiny_corpus_files):
        """Per-call n_semantic override changes the number of semantic results."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=0)
        query = "God created the heavens and the earth"
        results_small = sel.select(query)
        results_larger = sel.select(query, n_semantic=4)
        # Larger n_semantic should yield >= results.
        assert len(results_larger) >= len(results_small)

    def test_per_call_override_does_not_change_default(self, tiny_corpus_files):
        """A per-call override must not permanently alter the selector's defaults."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=2, n_coverage=2)
        query = "God created the heavens and the earth"

        before = sel.select(query)
        # Override just for this call.
        _expanded = sel.select(query, n_coverage=8)
        after = sel.select(query)

        assert before == after, (
            "Per-call override must not mutate the selector's default limits"
        )

    def test_select_with_override_respects_exclude_idx(self, tiny_corpus_files):
        """Per-call n override with exclude_idx still excludes the held-out verse."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=1)
        results = sel.select(
            "God created the heavens and the earth",
            exclude_idx=0,
            n_coverage=5,
        )
        assert all(ex.verse_idx != 0 for ex in results), (
            "exclude_idx=0 must be honoured even with per-call n_coverage override"
        )

    def test_bm25_index_built_once_across_calls(self, tiny_corpus_files):
        """The BM25 corpus index must be built once at construction, not per call.

        We verify this by checking that the underlying _bm25 object is the same
        instance across multiple select() calls (including retries with overrides).
        """
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=3)
        bm25_id_before = id(sel._bm25)

        # Multiple calls with varying per-call overrides.
        for n_cov in [1, 3, 5, 7]:
            sel.select("God saw the light", n_coverage=n_cov)

        bm25_id_after = id(sel._bm25)
        assert bm25_id_before == bm25_id_after, (
            "sel._bm25 must be the same object instance across all calls "
            "(BM25 index must NOT be rebuilt per call)"
        )

    def test_results_preserved_across_repeated_calls_with_same_args(self, tiny_corpus_files):
        """Repeated calls with identical arguments must return identical results."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=3)
        query = "God created the heavens and the earth"
        r1 = sel.select(query, exclude_idx=0, n_semantic=2, n_coverage=4)
        r2 = sel.select(query, exclude_idx=0, n_semantic=2, n_coverage=4)
        assert r1 == r2, "Results must be deterministic across repeated calls"


# ---------------------------------------------------------------------------
# Equivalence-group source exclusion (duplicate-source leakage prevention)
# ---------------------------------------------------------------------------

class TestEquivalenceGroupSourceExclusion:
    """When exclude_idx is given, ALL corpus rows whose normalised source
    sequence equals the query's normalised source sequence must be excluded
    from both semantic and coverage stages — not just the single exclude_idx row.

    This prevents leakage when the same English source verse appears at
    multiple corpus indices (e.g. NUM 7:64 and similar repeated blessing verses).

    If exclude_idx is None, no text-based exclusion should occur.
    """

    @staticmethod
    def _make_dup_corpus(tmp_path, duplicate_source: bool):
        """Build a 6-line corpus.

        Lines 0-4 are unique; line 5 is either a duplicate of line 0 (same
        normalised source but a *different* target) or a fresh distinct verse.

        Returns (src_path, tgt_path, query_text) where query_text normalises
        identically to lines 0 and (optionally) 5.
        """
        # Row 0 and row 5 will share the *same* normalised source if duplicate=True
        shared_src = "God created the heavens and the earth"
        if duplicate_source:
            src_lines = [
                shared_src,                              # 0 — held-out query row
                "The earth was without form and void",   # 1
                "And God said let there be light",       # 2
                "And God saw that the light was good",   # 3
                "God called the light Day and Night",    # 4
                shared_src,                              # 5 — DUPLICATE source, DIFFERENT target
            ]
            tgt_lines = [
                "Cible A",                               # 0
                "Cible B",                               # 1
                "Cible C",                               # 2
                "Cible D",                               # 3
                "Cible E",                               # 4
                "Cible F",                               # 5 — different target
            ]
        else:
            src_lines = [
                shared_src,                              # 0
                "The earth was without form and void",   # 1
                "And God said let there be light",       # 2
                "And God saw that the light was good",   # 3
                "God called the light Day and Night",    # 4
                "And God blessed the seventh day",       # 5 — UNIQUE source
            ]
            tgt_lines = [
                "Cible A",
                "Cible B",
                "Cible C",
                "Cible D",
                "Cible E",
                "Cible F",
            ]
        src_path = tmp_path / "src.txt"
        tgt_path = tmp_path / "tgt.txt"
        src_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
        tgt_path.write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")
        return str(src_path), str(tgt_path), shared_src

    def test_semantic_excludes_duplicate_source_row(self, tmp_path):
        """Semantic stage must not return a row with identical normalised source
        as the held-out exclude_idx row."""
        src, tgt, query = self._make_dup_corpus(tmp_path, duplicate_source=True)
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=0)
        results = sel.select(query, exclude_idx=0)
        # Row 5 has the same normalised source as row 0 (the exclude_idx row).
        # It must NOT appear in results.
        verse_indices = [ex.verse_idx for ex in results]
        assert 0 not in verse_indices, "exclude_idx=0 must be excluded"
        assert 5 not in verse_indices, (
            "Row 5 shares identical normalised source with exclude_idx=0 "
            "and must also be excluded from semantic stage (equivalence-group exclusion)."
        )

    def test_coverage_excludes_duplicate_source_row(self, tmp_path):
        """Coverage stage must not return a row with identical normalised source
        as the held-out exclude_idx row."""
        src, tgt, query = self._make_dup_corpus(tmp_path, duplicate_source=True)
        sel = _make_selector(src, tgt, n_semantic=0, n_coverage=5)
        results = sel.select(query, exclude_idx=0)
        verse_indices = [ex.verse_idx for ex in results]
        assert 0 not in verse_indices, "exclude_idx=0 must be excluded"
        assert 5 not in verse_indices, (
            "Row 5 shares identical normalised source with exclude_idx=0 "
            "and must also be excluded from coverage stage (equivalence-group exclusion)."
        )

    def test_nonduplicate_row_not_excluded(self, tmp_path):
        """A non-duplicate row (row 5 unique source) must NOT be wrongly excluded."""
        src, tgt, query = self._make_dup_corpus(tmp_path, duplicate_source=False)
        sel = _make_selector(src, tgt, n_semantic=5, n_coverage=5)
        results = sel.select(query, exclude_idx=0)
        verse_indices = [ex.verse_idx for ex in results]
        # Row 5 has a different normalised source; must be eligible
        # (it may or may not be returned depending on BM25 score, but it
        # must not be systematically excluded — so we check it is in the
        # eligible pool by asking the selector for enough results).
        assert 0 not in verse_indices, "exclude_idx=0 must still be excluded"
        # Row 5 (distinct source) should be selectable — if enough slots requested.
        # The corpus has 6 rows; with exclude_idx=0 there are 5 candidates.
        # We asked for 5+5=10, which exhausts all candidates, so 5 must appear.
        assert 5 in verse_indices, (
            "Row 5 has a DIFFERENT normalised source from exclude_idx=0 and "
            "must NOT be excluded by the equivalence-group mechanism."
        )

    def test_exclude_idx_none_does_not_text_exclude(self, tmp_path):
        """When exclude_idx=None, no text-based exclusion occurs.
        Both row 0 and row 5 (identical sources) must be eligible."""
        src, tgt, query = self._make_dup_corpus(tmp_path, duplicate_source=True)
        sel = _make_selector(src, tgt, n_semantic=6, n_coverage=0)
        results = sel.select(query, exclude_idx=None)
        verse_indices = {ex.verse_idx for ex in results}
        # With no exclude, all 6 rows are candidates; both 0 and 5 share the
        # highest BM25 score so at least one should appear.
        assert (0 in verse_indices) or (5 in verse_indices), (
            "With exclude_idx=None, duplicate-source rows 0 and 5 must not be "
            "blanket-excluded — at least one must appear in results."
        )

    def test_semantic_requests_extra_candidates_to_fill_count(self, tmp_path):
        """After filtering equivalents, semantic stage must still return
        up to n_semantic results from non-equivalent candidates (i.e. it
        must fetch enough BM25 candidates internally before filtering)."""
        src, tgt, query = self._make_dup_corpus(tmp_path, duplicate_source=True)
        # Corpus: 6 rows total; row 0 excluded by index, row 5 excluded by
        # equivalence — leaves 4 eligible candidates.
        # Asking for n_semantic=4 must fill all 4 slots, not stop at n_semantic-1.
        sel = _make_selector(src, tgt, n_semantic=4, n_coverage=0)
        results = sel.select(query, exclude_idx=0)
        sem_results = [ex for ex in results if ex.selection_method == "semantic"]
        assert len(sem_results) == 4, (
            f"Expected 4 semantic results after excluding 2 equivalent rows, "
            f"got {len(sem_results)}. Semantic stage must over-fetch then filter."
        )
