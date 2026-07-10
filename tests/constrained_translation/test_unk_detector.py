"""Tests for constrained_translation.unk_detector — Task 5.

TDD: all tests are written before the implementation.  Run them first to
confirm they fail (ImportError / AttributeError), then implement
unk_detector.py to make them pass.

The UNKDetector is source-side only.  It never calls the model, never
reads target vocabulary to infer source coverage, and never generates
target output or [UNK:…] markers.  It only identifies source spans that
lack a lexical match in the attested vocabulary.
"""

from __future__ import annotations

import unicodedata

import pytest

from constrained_translation.protocol import UNKSpan

# ---------------------------------------------------------------------------
# Import under test — will fail until unk_detector.py is created.
# ---------------------------------------------------------------------------
from constrained_translation.unk_detector import UNKDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vocab(*words: str) -> frozenset[str]:
    """Build a frozenset of attested vocabulary tokens from plain strings."""
    return frozenset(words)


# ---------------------------------------------------------------------------
# Test 1 — All tokens covered → no UNK spans returned
# ---------------------------------------------------------------------------

def test_unk_detector_no_spans_when_all_covered():
    """If every source token normalises to a form present in attested_vocab,
    the detector returns an empty tuple."""
    vocab = _vocab("In", "the", "beginning", "God", "created")
    source = "In the beginning God created"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert spans == ()


# ---------------------------------------------------------------------------
# Test 2 — Single uncovered token → one UNKSpan
# ---------------------------------------------------------------------------

def test_unk_detector_single_uncovered_token():
    """A single token absent from attested_vocab produces exactly one UNKSpan."""
    vocab = _vocab("In", "the", "God", "created")
    # "beginning" is NOT in vocab
    source = "In the beginning God created"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    assert spans[0].surface == "beginning"


# ---------------------------------------------------------------------------
# Test 3 — Adjacent uncovered tokens are merged into a single span
# ---------------------------------------------------------------------------

def test_unk_detector_adjacent_uncovered_tokens_merged():
    """Consecutive uncovered tokens are merged into a single UNKSpan."""
    vocab = _vocab("In", "God", "created")
    # "the" and "beginning" are both uncovered and adjacent → merged
    source = "In the beginning God created"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    assert spans[0].surface == "the beginning"


# ---------------------------------------------------------------------------
# Test 4 — Character offsets are correct
# ---------------------------------------------------------------------------

def test_unk_detector_character_offsets_are_correct():
    """start_char and end_char point to the correct slice of source_text."""
    # "selah" is not covered; it starts at char 12 in "and rejoice selah here"
    source = "and rejoice selah here"
    vocab = _vocab("and", "rejoice", "here")
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    span = spans[0]
    # Verify the slice reconstructs the surface text
    assert source[span.start_char:span.end_char] == span.surface
    assert span.surface == "selah"
    assert span.start_char == source.index("selah")
    assert span.end_char == span.start_char + len("selah")


def test_unk_detector_character_offsets_multi_span():
    """Offsets are correct for multiple disjoint UNK spans."""
    # "alpha" and "omega" are both uncovered; they are separated by "and"
    source = "alpha and omega"
    vocab = _vocab("and")
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 2
    for span in spans:
        assert source[span.start_char:span.end_char] == span.surface


# ---------------------------------------------------------------------------
# Test 5 — Surface text preserves original (non-normalised) form
# ---------------------------------------------------------------------------

def test_unk_detector_surface_text_is_original_not_normalised():
    """The UNKSpan.surface field holds the original source text, not the
    NFKC-lowercased normalised form used for lookup."""
    # "Selah" (capital S) is not in vocab even case-insensitively
    # surface must be "Selah", not "selah"
    source = "praise Selah forever"
    vocab = _vocab("praise", "forever")
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    # The original capitalisation must be preserved
    assert spans[0].surface == "Selah"


def test_unk_detector_normalised_form_used_for_lookup():
    """Coverage is determined by NFKC-lowercased comparison, so a vocab
    entry like 'Dieu' covers source token 'dieu' and vice-versa."""
    # vocab has "Dieu" (capital D); source has "dieu" (lowercase)
    # After normalise+lower both become "dieu" → token is covered
    vocab = _vocab("Dieu", "créa", "les")
    source = "dieu créa les cieux"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    # "cieux" is not in vocab → one span
    assert len(spans) == 1
    assert spans[0].surface == "cieux"


# ---------------------------------------------------------------------------
# Test 6 — Empty source text → empty tuple
# ---------------------------------------------------------------------------

def test_unk_detector_empty_source_returns_empty():
    """An empty source string produces no UNK spans."""
    vocab = _vocab("anything")
    detector = UNKDetector()
    spans = detector.detect("", vocab)
    assert spans == ()


def test_unk_detector_whitespace_only_source_returns_empty():
    """A source string containing only whitespace produces no UNK spans."""
    vocab = _vocab("anything")
    detector = UNKDetector()
    spans = detector.detect("   \t\n  ", vocab)
    assert spans == ()


# ---------------------------------------------------------------------------
# Test 7 — Determinism: same call returns identical output
# ---------------------------------------------------------------------------

def test_unk_detector_is_deterministic():
    """Calling detect() twice with identical inputs returns identical output."""
    vocab = _vocab("In", "God", "created")
    source = "In the beginning God created"
    detector = UNKDetector()
    spans_a = detector.detect(source, vocab)
    spans_b = detector.detect(source, vocab)
    assert spans_a == spans_b


def test_unk_detector_is_deterministic_across_instances():
    """Two separate UNKDetector instances return identical output."""
    vocab = _vocab("In", "God", "created")
    source = "In the beginning God created"
    spans_a = UNKDetector().detect(source, vocab)
    spans_b = UNKDetector().detect(source, vocab)
    assert spans_a == spans_b


# ---------------------------------------------------------------------------
# Focused suite — Unicode / punctuation edge cases
# ---------------------------------------------------------------------------

def test_unk_detector_unicode_nfkc_equivalence():
    """NFKC normalisation means that compatibility-equivalent forms are
    treated as the same token for coverage purposes."""
    # NFKC normalises the fullwidth 'ａ' (U+FF41) to 'a'
    vocab = _vocab("ａ")   # fullwidth a in vocab
    source = "a b"          # regular 'a' in source; 'b' not covered
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    # 'a' (regular) should match 'ａ' (fullwidth) after NFKC norm+lower
    assert len(spans) == 1
    assert spans[0].surface == "b"


def test_unk_detector_punctuation_attached_to_token():
    """A source token like 'selah.' whose stripped normalised form is covered
    — here 'selah' is in vocab — should be treated as covered if the
    normalised+lowercased full form matches (or per-spec, exact match).

    Per spec §7.5 the normalised token is checked AS-IS (whitespace split
    only, no punctuation stripping on the source side).  So 'selah.' is a
    different token from 'selah' and is NOT covered if only 'selah' is in
    the vocab.
    """
    vocab = _vocab("praise", "selah", "forever")
    source = "praise selah. forever"
    # "selah." normalises to "selah." (NFKC changes nothing here) — lowercase
    # is "selah." which != "selah", so it should be uncovered
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    assert spans[0].surface == "selah."


def test_unk_detector_empty_vocab_marks_all_tokens():
    """If attested_vocab is empty, every source token is uncovered and
    adjacent tokens are merged into a single span."""
    vocab: frozenset[str] = frozenset()
    source = "alpha beta gamma"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 1
    assert spans[0].surface == "alpha beta gamma"
    assert spans[0].start_char == 0
    assert spans[0].end_char == len(source)


def test_unk_detector_multiple_disjoint_spans():
    """Non-adjacent uncovered tokens produce separate UNKSpans."""
    vocab = _vocab("and")
    source = "alpha and omega"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    assert len(spans) == 2
    assert spans[0].surface == "alpha"
    assert spans[1].surface == "omega"


def test_unk_detector_does_not_read_target_vocab():
    """UNKDetector must work purely from source_text and attested_vocab;
    it never calls a model, never generates output, and returns only
    UNKSpan records.  This test asserts the return type contract."""
    vocab = _vocab("word")
    source = "word unknown"
    detector = UNKDetector()
    result = detector.detect(source, vocab)
    assert isinstance(result, tuple)
    for span in result:
        assert isinstance(span, UNKSpan)
        assert isinstance(span.surface, str)
        assert isinstance(span.start_char, int)
        assert isinstance(span.end_char, int)
        assert span.end_char > span.start_char


def test_unk_detector_result_is_tuple_of_unkspan():
    """detect() always returns a tuple[UNKSpan, ...], never a list."""
    vocab = _vocab("hello")
    source = "hello world"
    detector = UNKDetector()
    result = detector.detect(source, vocab)
    assert isinstance(result, tuple)


def test_unk_detector_spans_ordered_by_start_char():
    """Returned UNKSpans are in ascending start_char order."""
    vocab = _vocab("and", "the")
    source = "alpha and beta the gamma"
    detector = UNKDetector()
    spans = detector.detect(source, vocab)
    start_chars = [s.start_char for s in spans]
    assert start_chars == sorted(start_chars)
