"""Regression tests: unified source-unit normalizer handoff invariant.

The key invariant being tested:

    If ExampleSelector.select() selects examples whose source sides cover all
    normalised query units, then BatchRunner._check_source_coverage (which
    calls UNKDetector.detect with a vocab built from the same normalizer)
    must return NO UNK spans.

    Equivalently: if the selector's computed coverage says "all units covered",
    the detector must agree.

Before the fix:
  - ExampleSelector used BM25Query._normalize_text  (NFKC + lower + strip punct)
  - BatchRunner built source_vocab with NFKC only    (no lower, no punct strip)
  - UNKDetector compared with NFKC + lower           (no punct strip)

  Result: "you,," normalised in the selector to "you" (covered), but the
  detector saw "you,," != any vocab entry because vocab had "you," (NFKC only)
  and "you,," != "you," → bogus UNK, retries wasted.

After the fix all three components use normalize_source_units() which does
NFKC + lower + strip punct + collapse whitespace.
"""

from __future__ import annotations

import unicodedata

import pytest

from constrained_translation.example_selector import ExampleSelector
from constrained_translation.text_normalize import normalize_source_units
from constrained_translation.unk_detector import UNKDetector
from constrained_translation.protocol import AlignedExample


# ---------------------------------------------------------------------------
# normalize_source_units unit tests
# ---------------------------------------------------------------------------

class TestNormalizeSourceUnits:
    """Verify the shared helper directly."""

    def test_plain_text(self):
        assert normalize_source_units("Hello World") == ["hello", "world"]

    def test_nfkc_applied(self):
        # fullwidth 'ａ' (U+FF41) → 'a' under NFKC
        assert normalize_source_units("ａ") == ["a"]

    def test_lowercase(self):
        assert normalize_source_units("God Created") == ["god", "created"]

    def test_strip_trailing_comma(self):
        assert normalize_source_units("you,,") == ["you"]

    def test_strip_quotes(self):
        assert normalize_source_units('"word"') == ["word"]

    def test_strip_apostrophe(self):
        assert normalize_source_units("don't") == ["dont"]

    def test_strip_hyphen(self):
        assert normalize_source_units("well-known") == ["wellknown"]

    def test_punctuation_only_is_empty(self):
        assert normalize_source_units(",") == []
        assert normalize_source_units("...") == []
        assert normalize_source_units("!?") == []

    def test_empty_string(self):
        assert normalize_source_units("") == []

    def test_whitespace_only(self):
        assert normalize_source_units("   ") == []

    def test_sentence_with_punctuation(self):
        result = normalize_source_units("In the beginning, God created.")
        assert result == ["in", "the", "beginning", "god", "created"]

    def test_collapse_whitespace(self):
        assert normalize_source_units("a  b   c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Selector → Detector handoff invariant tests
# ---------------------------------------------------------------------------

def _make_aligned_example(verse_idx: int, source: str, target: str) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source=source,
        target=target,
        selection_score=1.0,
        selection_method="coverage",
        evidence_tier=1,
    )


class TestSelectorDetectorHandoffInvariant:
    """If selector's sources cover all normalised query units, UNKDetector returns no spans."""

    def test_basic_handoff_invariant(self):
        """Core invariant: when example sources cover all query units, no UNK spans."""
        query = "God created the earth"
        query_units = set(normalize_source_units(query))

        # Build examples that collectively cover all query units
        examples = (
            _make_aligned_example(0, "God created the heavens", "..."),
            _make_aligned_example(1, "And the earth was void", "..."),
        )

        # Verify selector-side coverage computation agrees
        covered: set[str] = set()
        for ex in examples:
            covered |= set(normalize_source_units(ex.source))
        assert query_units <= covered, "Test setup: examples must cover all query units"

        # Build source_vocab exactly as BatchRunner does (using normalize_source_units)
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        # UNKDetector must report no spans
        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert spans == (), (
            f"UNKDetector reported spans {spans!r} even though all query units "
            f"{query_units!r} are covered by vocab {source_vocab!r}"
        )

    def test_handoff_invariant_with_punctuation(self):
        """Regression: punctuation-attached tokens must not cause false UNK spans.

        If the query contains 'you,,' and examples contain 'you', the normalizer
        strips the commas from both sides so coverage is correctly determined.
        """
        query = "you,, love"
        query_units = set(normalize_source_units(query))  # {'you', 'love'}
        assert "you" in query_units, "Test setup: 'you' must be a query unit"

        examples = (
            _make_aligned_example(0, "you love the world", "..."),
        )

        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert spans == (), (
            f"Bogus UNK spans for punctuation-attached token: {[s.surface for s in spans]}"
        )

    def test_handoff_invariant_with_comma_separated_tokens(self):
        """'alpha,beta' as one whitespace-token strips comma → 'alphabeta' unit."""
        query = "alpha,beta"
        query_units = set(normalize_source_units(query))  # ['alphabeta']

        examples = (
            _make_aligned_example(0, "alphabeta gamma", "..."),
        )
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert spans == ()

    def test_handoff_invariant_quoted_word_covered(self):
        """'\"hello\"' in query is covered when 'hello' appears in example sources."""
        query = '"hello" world'
        query_units = set(normalize_source_units(query))  # {'hello', 'world'}

        examples = (
            _make_aligned_example(0, "hello world of light", "..."),
        )
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert spans == ()

    def test_handoff_invariant_hyphenated_covered(self):
        """'well-known' in query covered when 'wellknown' in example source."""
        query = "well-known theory"
        query_units = set(normalize_source_units(query))  # {'wellknown', 'theory'}

        examples = (
            _make_aligned_example(0, "wellknown theory exists", "..."),
        )
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert spans == ()

    def test_uncovered_unit_still_flagged(self):
        """When a query unit is genuinely absent from all examples, UNK is reported."""
        query = "god xyzzy earth"
        examples = (
            _make_aligned_example(0, "god and earth were created", "..."),
        )
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        assert len(spans) == 1
        assert "xyzzy" in spans[0].surface

    def test_standalone_punctuation_token_not_unk(self):
        """A standalone ',' token in the query is transparent — never flagged as UNK."""
        query = "alpha , omega"
        examples = (
            _make_aligned_example(0, "alpha and omega", "..."),
        )
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)
        # ',' is punctuation-only → transparent.  All real words are covered.
        assert spans == ()


# ---------------------------------------------------------------------------
# Integration test: full ExampleSelector → UNKDetector pipeline
# ---------------------------------------------------------------------------

class TestSelectorToDetectorIntegration:
    """Full pipeline: ExampleSelector + UNKDetector using the tiny corpus fixture."""

    def test_fully_covered_query_produces_no_unk_spans(self, tiny_corpus_files):
        """When ExampleSelector covers all query units, UNKDetector must agree.

        This exercises the complete selector→detector pipeline with a real
        BM25-backed corpus.
        """
        src, tgt = tiny_corpus_files
        selector = ExampleSelector(
            source_file=src,
            target_file=tgt,
            n_semantic=5,
            n_coverage=5,
        )

        # Use a query composed only of words that appear in the tiny corpus.
        query = "God created the earth"
        examples = selector.select(query, exclude_idx=None)

        # Compute what the selector considers covered
        query_units = set(normalize_source_units(query))
        covered_by_examples: set[str] = set()
        for ex in examples:
            covered_by_examples |= set(normalize_source_units(ex.source))

        if not (query_units <= covered_by_examples):
            pytest.skip(
                f"Corpus didn't cover all query units "
                f"(uncovered: {query_units - covered_by_examples}). "
                "Cannot test the handoff invariant."
            )

        # Build source vocab exactly as BatchRunner does
        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)

        assert spans == (), (
            f"Selector said all units were covered but UNKDetector found: "
            f"{[s.surface for s in spans]!r}\n"
            f"query_units={query_units!r}\n"
            f"source_vocab (sample)={sorted(source_vocab)[:20]!r}"
        )

    def test_selector_coverage_and_detector_agree_with_punctuated_query(
        self, tiny_corpus_files
    ):
        """Invariant holds even when query has trailing punctuation."""
        src, tgt = tiny_corpus_files
        selector = ExampleSelector(
            source_file=src,
            target_file=tgt,
            n_semantic=5,
            n_coverage=5,
        )

        # Include trailing punctuation in query
        query = "God, created the earth."
        examples = selector.select(query, exclude_idx=None)

        query_units = set(normalize_source_units(query))
        covered_by_examples: set[str] = set()
        for ex in examples:
            covered_by_examples |= set(normalize_source_units(ex.source))

        if not (query_units <= covered_by_examples):
            pytest.skip(
                f"Corpus didn't cover all query units "
                f"(uncovered: {query_units - covered_by_examples})."
            )

        source_vocab: frozenset[str] = frozenset(
            unit
            for ex in examples
            for unit in normalize_source_units(ex.source)
        )

        detector = UNKDetector()
        spans = detector.detect(query, source_vocab)

        assert spans == (), (
            f"Handoff mismatch with punctuated query: "
            f"{[s.surface for s in spans]!r}"
        )
