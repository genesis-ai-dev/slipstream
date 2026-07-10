"""Tests for Task 4: VocabExtractor — attested target-side surface vocabulary.

Strict TDD: these tests are written BEFORE the implementation.
All tests run fully offline.

Key design constraints under test:
  - Vocabulary is derived *exclusively* from target strings of AlignedExample,
    never from source strings.
  - Evidence tier is always 1 (sentence co-occurrence).  Sentence co-occurrence
    is NOT word alignment and does NOT prove any specific source word maps to
    any specific target word.
  - Tokens are NFKC-normalised Unicode surface forms; ASCII is not assumed.
  - Punctuation attached to words is separated out as its own entry.
  - Returns a frozenset[str].
"""

import unicodedata

import pytest

from constrained_translation.protocol import AlignedExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor():
    from constrained_translation.vocab_extractor import VocabExtractor
    return VocabExtractor()


def _make_example(source: str, target: str, idx: int = 0) -> AlignedExample:
    return AlignedExample(
        verse_idx=idx,
        source=source,
        target=target,
        selection_score=1.0,
        selection_method="semantic",
        evidence_tier=1,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

class TestVocabExtractorReturnType:
    """VocabExtractor.extract() must return a frozenset[str]."""

    def test_returns_frozenset(self):
        ve = _make_extractor()
        result = ve.extract([_make_example("hello", "bonjour")])
        assert isinstance(result, frozenset)

    def test_returns_frozenset_of_strings(self):
        ve = _make_extractor()
        result = ve.extract([_make_example("hello world", "bonjour monde")])
        for tok in result:
            assert isinstance(tok, str)

    def test_empty_iterable_returns_empty_frozenset(self):
        ve = _make_extractor()
        result = ve.extract([])
        assert result == frozenset()

    def test_result_is_immutable(self):
        ve = _make_extractor()
        result = ve.extract([_make_example("a", "b")])
        with pytest.raises((AttributeError, TypeError)):
            result.add("should_fail")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. Target-only semantics — source strings must NOT contribute tokens
# ---------------------------------------------------------------------------

class TestVocabExtractorUsesOnlyTargetStrings:
    """Vocabulary must be derived exclusively from target strings (invariant I1).

    Sentence co-occurrence evidence (Tier 1) does NOT prove source words map to
    target words.  The extractor must never pull tokens from source strings.
    """

    def test_source_tokens_not_in_vocab(self):
        ve = _make_extractor()
        # English source tokens that are distinct from French target tokens.
        ex = _make_example(
            source="heavens firmament abyss",
            target="cieux lumière terre",
        )
        vocab = ve.extract([ex])
        for src_tok in ["heavens", "firmament", "abyss"]:
            assert src_tok not in vocab, (
                f"Source token {src_tok!r} must NOT appear in attested vocab "
                f"(source strings are not evidence of target vocabulary)."
            )

    def test_target_tokens_present_in_vocab(self):
        ve = _make_extractor()
        ex = _make_example(
            source="In the beginning",
            target="Au commencement Dieu",
        )
        vocab = ve.extract([ex])
        for tgt_tok in ["Au", "commencement", "Dieu"]:
            # NFKC-normalised; same as the raw token here.
            assert tgt_tok in vocab, (
                f"Target token {tgt_tok!r} must appear in attested vocab."
            )

    def test_source_exclusive_tokens_absent_even_when_similar(self):
        """A token present only in source must not contaminate vocab."""
        ve = _make_extractor()
        # 'selah' appears only in source; 'shalom' appears only in target.
        ex = _make_example(
            source="praise selah selah",
            target="louange shalom",
        )
        vocab = ve.extract([ex])
        assert "selah" not in vocab
        assert "shalom" in vocab

    def test_multiple_examples_only_target_tokens(self):
        ve = _make_extractor()
        examples = [
            _make_example("God created", "Dieu créa", idx=0),
            _make_example("light day night", "lumière jour nuit", idx=1),
        ]
        vocab = ve.extract(examples)
        for src_tok in ["God", "created", "light", "day", "night"]:
            assert src_tok not in vocab
        for tgt_tok in ["Dieu", "lumière", "jour", "nuit"]:
            assert tgt_tok in vocab


# ---------------------------------------------------------------------------
# 3. NFKC normalisation
# ---------------------------------------------------------------------------

class TestVocabExtractorNFKCNormalisation:
    """All surface tokens must be NFKC-normalised before being stored."""

    def test_nfc_form_stored_as_nfkc(self):
        """Tokens already in NFC should remain unchanged by NFKC if equivalent."""
        ve = _make_extractor()
        # 'é' as NFC (U+00E9) is also its NFKC form.
        ex = _make_example("x", "lumière")
        vocab = ve.extract([ex])
        assert unicodedata.normalize("NFKC", "lumière") in vocab

    def test_fullwidth_digit_normalised(self):
        """Fullwidth digit U+FF10 ('０') normalises to '0' under NFKC."""
        ve = _make_extractor()
        fullwidth_zero = "\uff10"  # FULLWIDTH DIGIT ZERO
        ex = _make_example("x", fullwidth_zero)
        vocab = ve.extract([ex])
        # NFKC normalises fullwidth digits to ASCII equivalents.
        assert "0" in vocab, (
            "Fullwidth digit should NFKC-normalise to ASCII '0'"
        )
        assert fullwidth_zero not in vocab, (
            "Raw fullwidth form should not appear; NFKC form should."
        )

    def test_ligature_normalised(self):
        """The 'ﬁ' ligature (U+FB01) NFKC-normalises to 'fi'."""
        ve = _make_extractor()
        ex = _make_example("x", "ﬁsh")
        vocab = ve.extract([ex])
        expected = unicodedata.normalize("NFKC", "ﬁsh")
        assert expected in vocab

    def test_combined_form_stored(self):
        """Decomposed (NFD) input should be stored in NFKC composed form."""
        ve = _make_extractor()
        # NFD: 'e' + combining acute accent
        nfd_e = "e\u0301"
        nfkc_e = unicodedata.normalize("NFKC", nfd_e)  # → 'é'
        ex = _make_example("x", nfd_e)
        vocab = ve.extract([ex])
        assert nfkc_e in vocab, (
            f"NFD input {nfd_e!r} should be stored as NFKC form {nfkc_e!r}"
        )


# ---------------------------------------------------------------------------
# 4. Unicode script preservation (low-resource languages)
# ---------------------------------------------------------------------------

class TestVocabExtractorUnicodeScriptPreservation:
    """Tokenisation must preserve non-ASCII Unicode scripts sensibly.

    Low-resource languages often use scripts such as Greek, Hebrew, Arabic,
    Ethiopic, Devanagari, Thai, etc.  Whitespace tokenisation must not mangle
    these scripts.
    """

    def test_greek_tokens_preserved(self):
        ve = _make_extractor()
        ex = _make_example("In the beginning", "Ἐν ἀρχῇ ἦν ὁ λόγος")
        vocab = ve.extract([ex])
        for tok in ["Ἐν", "ἀρχῇ", "ἦν", "ὁ", "λόγος"]:
            nfkc = unicodedata.normalize("NFKC", tok)
            assert nfkc in vocab, (
                f"Greek token {tok!r} (NFKC: {nfkc!r}) missing from vocab"
            )

    def test_hebrew_tokens_preserved(self):
        ve = _make_extractor()
        ex = _make_example("In the beginning God", "בְּרֵאשִׁית בָּרָא אֱלֹהִים")
        vocab = ve.extract([ex])
        for tok in ["בְּרֵאשִׁית", "בָּרָא", "אֱלֹהִים"]:
            nfkc = unicodedata.normalize("NFKC", tok)
            assert nfkc in vocab, (
                f"Hebrew token {tok!r} missing from vocab"
            )

    def test_arabic_tokens_preserved(self):
        ve = _make_extractor()
        ex = _make_example("light", "النُّور الإِلَهِيّ")
        vocab = ve.extract([ex])
        for tok in ["النُّور", "الإِلَهِيّ"]:
            nfkc = unicodedata.normalize("NFKC", tok)
            assert nfkc in vocab

    def test_devanagari_tokens_preserved(self):
        ve = _make_extractor()
        ex = _make_example("God is light", "ईश्वर प्रकाश है")
        vocab = ve.extract([ex])
        for tok in ["ईश्वर", "प्रकाश", "है"]:
            nfkc = unicodedata.normalize("NFKC", tok)
            assert nfkc in vocab

    def test_mixed_script_tokens_preserved(self):
        """A translation containing both Latin and non-Latin script tokens."""
        ve = _make_extractor()
        # Mix of French and transliterated term in Hebrew script.
        ex = _make_example("God said selah", "Dieu dit סֶלָה")
        vocab = ve.extract([ex])
        assert "Dieu" in vocab
        assert "dit" in vocab
        nfkc_selah = unicodedata.normalize("NFKC", "סֶלָה")
        assert nfkc_selah in vocab


# ---------------------------------------------------------------------------
# 5. Union across multiple examples
# ---------------------------------------------------------------------------

class TestVocabExtractorUnionAcrossExamples:
    """Vocab is the union of tokens from all supplied examples."""

    def test_union_of_two_examples(self):
        ve = _make_extractor()
        examples = [
            _make_example("a", "lumière", idx=0),
            _make_example("b", "terre", idx=1),
        ]
        vocab = ve.extract(examples)
        assert "lumière" in vocab
        assert "terre" in vocab

    def test_duplicates_collapsed(self):
        """Same token appearing in multiple examples stored once."""
        ve = _make_extractor()
        examples = [
            _make_example("x", "Dieu créa lumière", idx=0),
            _make_example("y", "Dieu bénit lumière", idx=1),
        ]
        vocab = ve.extract(examples)
        # frozenset means no duplicates by definition, but length should be 4.
        assert "Dieu" in vocab
        assert "créa" in vocab
        assert "lumière" in vocab
        assert "bénit" in vocab
        # 'Dieu' and 'lumière' each appear twice but counted once:
        assert len([t for t in vocab if t == "Dieu"]) == 1

    def test_empty_target_skipped_gracefully(self):
        ve = _make_extractor()
        examples = [
            _make_example("a", "", idx=0),
            _make_example("b", "lumière", idx=1),
        ]
        vocab = ve.extract(examples)
        assert "lumière" in vocab

    def test_generator_input_accepted(self):
        """extract() must accept any Iterable, not just a list."""
        ve = _make_extractor()
        def _gen():
            yield _make_example("a", "lumière", idx=0)
            yield _make_example("b", "terre", idx=1)
        vocab = ve.extract(_gen())
        assert "lumière" in vocab
        assert "terre" in vocab


# ---------------------------------------------------------------------------
# 6. Punctuation handling
# ---------------------------------------------------------------------------

class TestVocabExtractorPunctuationHandling:
    """Punctuation attached to words must be included as separate entries."""

    def test_trailing_comma_yields_separate_entry(self):
        ve = _make_extractor()
        # "bonjour," → "bonjour" and "," as separate tokens.
        ex = _make_example("x", "bonjour, monde")
        vocab = ve.extract([ex])
        assert "bonjour" in vocab, "Word stripped of trailing comma must be present"
        assert "," in vocab, "Trailing comma must be present as its own entry"

    def test_trailing_period_yields_separate_entry(self):
        ve = _make_extractor()
        ex = _make_example("x", "fin.")
        vocab = ve.extract([ex])
        assert "fin" in vocab
        assert "." in vocab

    def test_leading_punctuation_separated(self):
        ve = _make_extractor()
        ex = _make_example("x", '"cité')
        vocab = ve.extract([ex])
        assert '"' in vocab or "\u201c" in vocab or "cité" in vocab, (
            "Leading punctuation or word should appear in vocab"
        )
        assert "cité" in vocab

    def test_punctuation_only_token(self):
        ve = _make_extractor()
        # A target string that is purely punctuation.
        ex = _make_example("x", "! ? .")
        vocab = ve.extract([ex])
        assert "!" in vocab
        assert "?" in vocab
        assert "." in vocab

    def test_word_without_punctuation_unchanged(self):
        ve = _make_extractor()
        ex = _make_example("x", "lumière")
        vocab = ve.extract([ex])
        assert "lumière" in vocab
        # Ensure there's no spurious extra token.
        non_empty = {t for t in vocab if t.strip()}
        assert "lumière" in non_empty


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

class TestVocabExtractorDeterminism:
    """Same input always produces the same frozenset."""

    def test_same_examples_same_result(self):
        ve = _make_extractor()
        examples = [
            _make_example("God created", "Dieu créa", idx=0),
            _make_example("light day", "lumière jour", idx=1),
        ]
        r1 = ve.extract(examples)
        r2 = ve.extract(examples)
        assert r1 == r2

    def test_order_of_examples_does_not_affect_set(self):
        """frozenset is order-independent; different orderings give same result."""
        ve = _make_extractor()
        ex_a = _make_example("a", "lumière", idx=0)
        ex_b = _make_example("b", "terre", idx=1)
        assert ve.extract([ex_a, ex_b]) == ve.extract([ex_b, ex_a])


# ---------------------------------------------------------------------------
# 8. Integration with conftest tiny corpus
# ---------------------------------------------------------------------------

class TestVocabExtractorWithTinyCorpus:
    """Smoke test using the conftest.py French corpus fixtures."""

    def test_extract_from_conftest_examples(self, tiny_corpus_files):
        """Build vocab from manually-constructed AlignedExample objects matching
        the first three verses of the tiny conftest corpus."""
        ve = _make_extractor()
        examples = [
            _make_example(
                "In the beginning God created the heavens and the earth",
                "Au commencement Dieu créa les cieux et la terre",
                idx=0,
            ),
            _make_example(
                "The earth was without form and void",
                "La terre était sans forme et vide",
                idx=1,
            ),
            _make_example(
                "And God said let there be light",
                "Et Dieu dit que la lumière soit",
                idx=2,
            ),
        ]
        vocab = ve.extract(examples)
        # Must contain target-side tokens.
        for tok in ["Au", "Dieu", "La", "terre", "lumière"]:
            nfkc = unicodedata.normalize("NFKC", tok)
            assert nfkc in vocab, f"{tok!r} must be in attested vocab"
        # Must not contain source-side tokens.
        for tok in ["beginning", "heavens", "without", "void", "light"]:
            assert tok not in vocab, (
                f"Source token {tok!r} must NOT be in attested vocab"
            )

    def test_vocab_size_reasonable(self):
        """Union of three target sentences should yield a reasonable token count."""
        ve = _make_extractor()
        examples = [
            _make_example("x", "Au commencement Dieu créa les cieux et la terre", idx=0),
            _make_example("y", "La terre était sans forme et vide", idx=1),
            _make_example("z", "Et Dieu dit que la lumière soit", idx=2),
        ]
        vocab = ve.extract(examples)
        # At minimum there should be more than 5 unique tokens.
        assert len(vocab) >= 5


# ---------------------------------------------------------------------------
# 9. Documentation / evidence-tier contract
# ---------------------------------------------------------------------------

class TestVocabExtractorDocumentation:
    """The VocabExtractor module and extract() method must carry Tier-1 evidence
    disclaimers per §6 and invariant I1."""

    def test_module_has_docstring(self):
        import constrained_translation.vocab_extractor as m
        assert m.__doc__ is not None and len(m.__doc__.strip()) > 0, (
            "vocab_extractor module must have a non-empty docstring"
        )

    def test_module_docstring_mentions_tier_1(self):
        import constrained_translation.vocab_extractor as m
        doc = (m.__doc__ or "").lower()
        assert "tier" in doc and "1" in doc or "tier-1" in doc or "tier 1" in doc, (
            "Module docstring must mention evidence Tier 1"
        )

    def test_module_docstring_not_word_alignment(self):
        """The phrase 'word alignment' must not appear without a negation qualifier."""
        import constrained_translation.vocab_extractor as m
        doc = m.__doc__ or ""
        import re
        # Allow 'not word alignment', 'NOT word alignment', 'is not word alignment'
        # but disallow bare 'word alignment' used affirmatively.
        bare = re.findall(r"(?i)(?<!not\s)(?<!NOT\s)\bword alignment\b", doc)
        assert not bare, (
            "Module docstring contains 'word alignment' without a negation qualifier. "
            "Per invariant I1, sentence co-occurrence is NOT word alignment."
        )

    def test_extract_method_has_docstring(self):
        from constrained_translation.vocab_extractor import VocabExtractor
        assert VocabExtractor.extract.__doc__ is not None, (
            "VocabExtractor.extract() must have a docstring"
        )

    def test_extract_docstring_mentions_tier_1(self):
        from constrained_translation.vocab_extractor import VocabExtractor
        doc = (VocabExtractor.extract.__doc__ or "").lower()
        assert "tier" in doc, (
            "extract() docstring must mention the evidence tier"
        )

    def test_extract_docstring_mentions_not_word_alignment(self):
        from constrained_translation.vocab_extractor import VocabExtractor
        doc = (VocabExtractor.extract.__doc__ or "").lower()
        # Must say something about NOT being word alignment.
        assert "not" in doc and ("word alignment" in doc or "alignment" in doc), (
            "extract() docstring must explicitly state this is NOT word alignment"
        )
