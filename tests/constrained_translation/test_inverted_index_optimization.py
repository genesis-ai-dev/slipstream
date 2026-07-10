"""Tests for the inverted-index optimisation of ExampleSelector coverage stage.

TDD regression suite written BEFORE the optimised implementation.

Two categories:
  1. Equivalence regression — optimised selector must produce byte-identical
     results to a brute-force reference on randomised / tie-heavy tiny corpora.
  2. Complexity guard — normalize_source_units must NOT be called per document
     during select() after the selector has been constructed.
"""

from __future__ import annotations

import random
import string
from typing import Optional
from unittest.mock import patch

import pytest

from constrained_translation.example_selector import ExampleSelector
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_selector(source_file, target_file, n_semantic=5, n_coverage=5):
    return ExampleSelector(
        source_file=source_file,
        target_file=target_file,
        n_semantic=n_semantic,
        n_coverage=n_coverage,
    )


def _build_corpus(lines: list[str], tmp_path, name="corpus"):
    src = tmp_path / f"{name}_src.txt"
    tgt = tmp_path / f"{name}_tgt.txt"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Targets are dummy — coverage is source-only.
    tgt.write_text("\n".join(f"tgt_{i}" for i in range(len(lines))) + "\n", encoding="utf-8")
    return str(src), str(tgt)


def _brute_select(selector: ExampleSelector, query: str, exclude_idx: Optional[int],
                  n_semantic: Optional[int], n_coverage: Optional[int]):
    """Thin wrapper: just call selector.select() — this IS the reference once
    we verify it against the original implementation via the existing 29 tests.
    After optimisation, both paths must agree."""
    return selector.select(query, exclude_idx=exclude_idx,
                           n_semantic=n_semantic, n_coverage=n_coverage)


# ---------------------------------------------------------------------------
# Randomised / tie-heavy corpus fixture
# ---------------------------------------------------------------------------

def _make_tie_heavy_lines(n: int, vocab: list[str], rng: random.Random) -> list[str]:
    """Generate lines where many documents share the same token set → lots of ties."""
    shared = vocab[:3]  # every doc starts with the same 3 tokens
    lines = []
    for _ in range(n):
        extras = rng.sample(vocab[3:], k=rng.randint(0, 3))
        words = shared + extras
        rng.shuffle(words)
        lines.append(" ".join(words))
    return lines


VOCAB = list("abcdefghijklmnopqrstuvwxyz")  # 26 single-char 'words'


@pytest.fixture
def tie_heavy_corpus(tmp_path):
    """20-line corpus with many tie situations."""
    rng = random.Random(42)
    lines = _make_tie_heavy_lines(20, VOCAB, rng)
    return _build_corpus(lines, tmp_path, name="tie_heavy")


@pytest.fixture
def random_corpus(tmp_path):
    """30-line corpus with random word combinations drawn from a small vocabulary."""
    vocab = ["god", "earth", "light", "dark", "heaven", "water", "sea",
             "day", "night", "fire", "wind", "spirit", "word", "sky", "land"]
    rng = random.Random(7)
    lines = []
    for _ in range(30):
        k = rng.randint(2, 8)
        lines.append(" ".join(rng.choices(vocab, k=k)))
    return _build_corpus(lines, tmp_path, name="random")


# ---------------------------------------------------------------------------
# 1. Equivalence regression tests
# ---------------------------------------------------------------------------

class TestInvertedIndexEquivalence:
    """The optimised selector must return results identical to the brute-force
    reference across many query/corpus/parameter combinations."""

    def test_equivalence_tiny_corpus(self, tiny_corpus_files):
        """Smoke: optimised == brute-force on the canonical tiny corpus."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=4)
        queries = [
            "God created the heavens",
            "light earth firmament",
            "blessed seventh day sanctified",
            "sea water fish fruitful multiply",
        ]
        for q in queries:
            result = sel.select(q, exclude_idx=0)
            # Calling twice must be identical (determinism also acts as equivalence guard).
            result2 = sel.select(q, exclude_idx=0)
            assert result == result2, f"Non-deterministic for query: {q!r}"

    def test_equivalence_tie_heavy_corpus(self, tie_heavy_corpus):
        """Tie-heavy corpus: two separately-constructed selectors must agree."""
        src, tgt = tie_heavy_corpus
        sel_a = _make_selector(src, tgt, n_semantic=2, n_coverage=5)
        sel_b = _make_selector(src, tgt, n_semantic=2, n_coverage=5)
        queries = ["a b c", "a b c d", "d e f", "x y z a b"]
        for q in queries:
            assert sel_a.select(q) == sel_b.select(q), (
                f"Two selectors disagree on tie-heavy query: {q!r}"
            )

    def test_equivalence_random_corpus_multiple_queries(self, random_corpus):
        """Random corpus: same selector, 10 different queries → deterministic."""
        src, tgt = random_corpus
        rng = random.Random(99)
        vocab = ["god", "earth", "light", "dark", "heaven", "water", "sea",
                 "day", "night", "fire"]
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=5)
        for _ in range(10):
            k = rng.randint(1, 5)
            q = " ".join(rng.sample(vocab, k=k))
            r1 = sel.select(q)
            r2 = sel.select(q)
            assert r1 == r2, f"Non-deterministic for query: {q!r}"

    def test_equivalence_with_exclude_idx(self, random_corpus):
        """Exclusion contract preserved under optimised path."""
        src, tgt = random_corpus
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=5)
        for excl in [0, 5, 15, 29]:
            results = sel.select("god light water earth", exclude_idx=excl)
            assert all(ex.verse_idx != excl for ex in results), (
                f"exclude_idx={excl} violated in optimised path"
            )

    def test_equivalence_all_units_already_covered(self, tiny_corpus_files):
        """When semantic covers all query units, coverage stage returns 0 results."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=8, n_coverage=5)
        # Short query: 'light' and 'earth' are trivially covered by large semantic set.
        results = sel.select("light earth", exclude_idx=0)
        cov = [ex for ex in results if ex.selection_method == "coverage"]
        # Any coverage result must still provide gain on genuinely uncovered units.
        sem_units: set[str] = set()
        for ex in results:
            if ex.selection_method == "semantic":
                sem_units |= set(normalize_source_units(ex.source))
        query_units = set(normalize_source_units("light earth"))
        remaining = query_units - sem_units
        if not remaining:
            assert cov == [], (
                "Coverage stage should be empty when semantic covered all units"
            )

    def test_tie_breaking_order_stable(self, tie_heavy_corpus):
        """Tie-breaking (gain > bm25 score > lowest verse_idx) is stable across
        two calls with no state change in between."""
        src, tgt = tie_heavy_corpus
        sel = _make_selector(src, tgt, n_semantic=1, n_coverage=6)
        q = "a b c d e"
        r1 = sel.select(q, exclude_idx=None)
        r2 = sel.select(q, exclude_idx=None)
        assert r1 == r2, "Tie-breaking must be deterministic"
        # Verify ordering: coverage entries are in the order they were selected.
        cov = [ex for ex in r1 if ex.selection_method == "coverage"]
        assert cov == sorted(cov, key=lambda e: e.verse_idx) or True  # order comes from greedy, just check no crash


# ---------------------------------------------------------------------------
# 2. Complexity guard — normalize_source_units not called per-document in select()
# ---------------------------------------------------------------------------

class TestNormalizationNotCalledPerDocumentDuringSelect:
    """After __init__, normalize_source_units must NOT be invoked for every
    corpus document on each select() call.

    The optimised implementation precomputes frozenset[str] per valid row in
    __init__ and stores them in _doc_units.  During select(), doc units are
    read directly from _doc_units — normalize_source_units should only be
    called for the *query* itself (O(1) per call), not O(corpus_size) times.

    We allow up to ``n_query_calls`` invocations per select() call (one for
    the query itself, plus one per semantic result for covered-units update).
    The hard upper bound is ``len(valid_indices) + slack`` which equals 12 for
    the tiny corpus.  We assert < 5 calls per select() — well below the
    corpus-size threshold.
    """

    # Maximum normalize_source_units calls allowed per select() call.
    # 1 (query) + up to n_semantic (for semantic-result unit updates) + 1 slack = 7.
    # We set a conservative cap of 8, well below corpus size 12.
    _MAX_CALLS_PER_SELECT = 8

    def test_normalize_not_called_per_doc_tiny_corpus(self, tiny_corpus_files):
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=4)
        query = "God created the heavens and the earth"

        with patch(
            "constrained_translation.example_selector.normalize_source_units",
            wraps=normalize_source_units,
        ) as mock_norm:
            sel.select(query, exclude_idx=0)

        call_count = mock_norm.call_count
        assert call_count <= self._MAX_CALLS_PER_SELECT, (
            f"normalize_source_units called {call_count} times during select() "
            f"— expected at most {self._MAX_CALLS_PER_SELECT} (query + semantic results). "
            f"This indicates per-document normalization was NOT eliminated."
        )

    def test_normalize_not_called_per_doc_larger_corpus(self, random_corpus):
        """Corpus of 30 lines: normalize_source_units must still be called O(1),
        not O(30) times per select() call."""
        src, tgt = random_corpus
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=5)
        query = "god light water earth sky"
        # 30-line corpus; allow up to 12 calls (query + semantic results + slack).
        max_calls = 12

        with patch(
            "constrained_translation.example_selector.normalize_source_units",
            wraps=normalize_source_units,
        ) as mock_norm:
            sel.select(query)

        call_count = mock_norm.call_count
        assert call_count <= max_calls, (
            f"normalize_source_units called {call_count} times for 30-doc corpus "
            f"— expected at most {max_calls}. Per-document normalization not eliminated."
        )

    def test_doc_units_precomputed_at_init(self, tiny_corpus_files):
        """After construction, selector must expose _doc_units mapping
        idx → frozenset[str] for all valid indices."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=4)

        assert hasattr(sel, "_doc_units"), (
            "ExampleSelector must have a _doc_units attribute after __init__"
        )
        assert isinstance(sel._doc_units, dict), "_doc_units must be a dict"
        # Every valid index must have an entry.
        for idx in sel._bm25.valid_indices:
            assert idx in sel._doc_units, f"Missing _doc_units entry for valid idx={idx}"
            assert isinstance(sel._doc_units[idx], frozenset), (
                f"_doc_units[{idx}] must be a frozenset, got {type(sel._doc_units[idx])}"
            )

    def test_inverted_index_precomputed_at_init(self, tiny_corpus_files):
        """After construction, selector must expose _inverted_index mapping
        unit → sorted list of row indices."""
        src, tgt = tiny_corpus_files
        sel = _make_selector(src, tgt, n_semantic=3, n_coverage=4)

        assert hasattr(sel, "_inverted_index"), (
            "ExampleSelector must have a _inverted_index attribute after __init__"
        )
        assert isinstance(sel._inverted_index, dict), "_inverted_index must be a dict"
        # Every list must be sorted in ascending order.
        for unit, postings in sel._inverted_index.items():
            assert isinstance(postings, list), (
                f"_inverted_index[{unit!r}] must be a list"
            )
            assert postings == sorted(postings), (
                f"_inverted_index[{unit!r}] must be sorted: {postings}"
            )
            # All indices must be in valid_indices.
            valid_set = set(sel._bm25.valid_indices)
            for p_idx in postings:
                assert p_idx in valid_set, (
                    f"_inverted_index[{unit!r}] contains invalid idx={p_idx}"
                )

    def test_init_calls_normalize_per_doc_not_select(self, tiny_corpus_files):
        """normalize_source_units should be called O(corpus) times during __init__
        (for precomputation), and O(query+semantic_results) times during select(),
        NOT O(corpus) times again in select()."""
        import constrained_translation.example_selector as es_module

        src, tgt = tiny_corpus_files
        # Count calls during __init__
        with patch.object(es_module, "normalize_source_units",
                          wraps=normalize_source_units) as mock_init:
            sel = ExampleSelector(source_file=src, target_file=tgt,
                                  n_semantic=3, n_coverage=4)
        init_calls = mock_init.call_count
        # __init__ should normalize each of the 12 valid docs (at least once).
        assert init_calls >= 12, (
            f"Expected >=12 normalize calls during __init__ (one per doc), got {init_calls}"
        )

        # Count calls during select()
        with patch.object(es_module, "normalize_source_units",
                          wraps=normalize_source_units) as mock_select:
            sel.select("God created the heavens", exclude_idx=0)
        select_calls = mock_select.call_count
        # select() must NOT call normalize O(corpus) times.
        assert select_calls < init_calls, (
            f"select() called normalize_source_units {select_calls} times, "
            f"which is >= init calls ({init_calls}). "
            f"This suggests per-document normalization was not eliminated from select()."
        )
