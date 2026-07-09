"""Tests for Task 2: protocol dataclasses, BackendProtocol, and FakeBackend.

Strict TDD: these tests are written before the implementation.
"""
import dataclasses
from typing import runtime_checkable

import pytest


# ---------------------------------------------------------------------------
# Protocol dataclass tests
# ---------------------------------------------------------------------------

class TestAlignedExample:
    def test_is_frozen_dataclass(self):
        from constrained_translation.protocol import AlignedExample
        ex = AlignedExample(
            verse_idx=0,
            source="In the beginning",
            target="Au commencement",
            selection_score=3.14,
            selection_method="semantic",
            evidence_tier=1,
        )
        assert dataclasses.is_dataclass(ex)
        # frozen: assigning should raise
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ex.verse_idx = 99  # type: ignore[misc]

    def test_fields(self):
        from constrained_translation.protocol import AlignedExample
        field_names = {f.name for f in dataclasses.fields(AlignedExample)}
        assert field_names == {
            "verse_idx",
            "source",
            "target",
            "selection_score",
            "selection_method",
            "evidence_tier",
        }

    def test_hashable(self):
        from constrained_translation.protocol import AlignedExample
        ex = AlignedExample(
            verse_idx=1,
            source="foo",
            target="bar",
            selection_score=1.0,
            selection_method="coverage",
            evidence_tier=1,
        )
        assert hash(ex) is not None
        s = {ex}
        assert ex in s


class TestUNKSpan:
    def test_is_frozen_dataclass(self):
        from constrained_translation.protocol import UNKSpan
        span = UNKSpan(surface="selah", start_char=12, end_char=17)
        assert dataclasses.is_dataclass(span)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            span.surface = "other"  # type: ignore[misc]

    def test_fields(self):
        from constrained_translation.protocol import UNKSpan
        field_names = {f.name for f in dataclasses.fields(UNKSpan)}
        assert field_names == {"surface", "start_char", "end_char"}

    def test_hashable(self):
        from constrained_translation.protocol import UNKSpan
        span = UNKSpan(surface="xyz", start_char=0, end_char=3)
        s = {span}
        assert span in s


class TestTranslationRequest:
    def test_is_mutable_dataclass(self):
        from constrained_translation.protocol import (
            AlignedExample,
            TranslationRequest,
            UNKSpan,
        )
        req = TranslationRequest(
            item_id="GEN 1:1",
            source_text="In the beginning",
            semantic_examples=(),
            coverage_examples=(),
            attested_vocab=frozenset(["foo"]),
            unk_spans=(),
            grammar_str="translation ::= \"foo\"",
            prompt="[Source]: In the beginning\n[Translation]:",
        )
        assert dataclasses.is_dataclass(req)
        # mutable — should not raise
        req.item_id = "GEN 1:2"
        assert req.item_id == "GEN 1:2"

    def test_fields(self):
        from constrained_translation.protocol import TranslationRequest
        field_names = {f.name for f in dataclasses.fields(TranslationRequest)}
        assert field_names == {
            "item_id",
            "source_text",
            "semantic_examples",
            "coverage_examples",
            "attested_vocab",
            "unk_spans",
            "grammar_str",
            "prompt",
        }


class TestGenerationResult:
    def test_is_mutable_dataclass(self):
        from constrained_translation.protocol import GenerationResult
        gr = GenerationResult(
            text="Au commencement",
            token_ids=[1, 2, 3],
            prompt_tokens=10,
            output_tokens=3,
            generation_ms=42.0,
        )
        assert dataclasses.is_dataclass(gr)
        gr.text = "changed"
        assert gr.text == "changed"

    def test_fields(self):
        from constrained_translation.protocol import GenerationResult
        field_names = {f.name for f in dataclasses.fields(GenerationResult)}
        assert field_names == {
            "text",
            "token_ids",
            "prompt_tokens",
            "output_tokens",
            "generation_ms",
        }


class TestTokenAuditResult:
    def test_is_mutable_dataclass(self):
        from constrained_translation.protocol import TokenAuditResult
        tar = TokenAuditResult(passed=True, violations=[])
        assert dataclasses.is_dataclass(tar)
        tar.passed = False

    def test_fields(self):
        from constrained_translation.protocol import TokenAuditResult
        field_names = {f.name for f in dataclasses.fields(TokenAuditResult)}
        assert field_names == {"passed", "violations"}


class TestTranslationResult:
    def test_is_mutable_dataclass(self):
        from constrained_translation.protocol import (
            GenerationResult,
            TranslationRequest,
            TranslationResult,
        )
        req = TranslationRequest(
            item_id="GEN 1:1",
            source_text="foo",
            semantic_examples=(),
            coverage_examples=(),
            attested_vocab=frozenset(),
            unk_spans=(),
            grammar_str="",
            prompt="",
        )
        tr = TranslationResult(
            item_id="GEN 1:1",
            source_text="foo",
            translation="bar",
            coverage_pass=True,
            retry_count=0,
            hard_failure=False,
            generation_result=None,
            token_audit=None,
            request=req,
            error=None,
        )
        assert dataclasses.is_dataclass(tr)

    def test_fields(self):
        from constrained_translation.protocol import TranslationResult
        field_names = {f.name for f in dataclasses.fields(TranslationResult)}
        assert field_names == {
            "item_id",
            "source_text",
            "translation",
            "coverage_pass",
            "retry_count",
            "hard_failure",
            "generation_result",
            "token_audit",
            "request",
            "error",
        }


class TestBatchRollup:
    def test_is_mutable_dataclass(self):
        from constrained_translation.protocol import BatchRollup
        br = BatchRollup(
            total_units=10,
            mean_tok_per_sec=100.0,
            coverage_pass_pct=90.0,
            retry_pct=5.0,
            hard_failure_pct=1.0,
            token_audit_pass_pct=95.0,
            total_input_tokens=500,
            total_output_tokens=200,
        )
        assert dataclasses.is_dataclass(br)

    def test_fields(self):
        from constrained_translation.protocol import BatchRollup
        field_names = {f.name for f in dataclasses.fields(BatchRollup)}
        assert field_names == {
            "total_units",
            "mean_tok_per_sec",
            "coverage_pass_pct",
            "retry_pct",
            "hard_failure_pct",
            "token_audit_pass_pct",
            "total_input_tokens",
            "total_output_tokens",
        }


# ---------------------------------------------------------------------------
# BackendProtocol tests
# ---------------------------------------------------------------------------

class TestBackendProtocol:
    def test_is_runtime_checkable(self):
        """BackendProtocol must be decorated with @runtime_checkable."""
        from constrained_translation.protocol import BackendProtocol
        # Can use isinstance() at runtime
        assert hasattr(BackendProtocol, "_is_protocol")

    def test_protocol_methods(self):
        """Protocol defines the four required methods."""
        import inspect
        from constrained_translation.protocol import BackendProtocol
        members = {name for name, _ in inspect.getmembers(BackendProtocol)}
        assert "generate" in members
        assert "tokenize" in members
        assert "decode_token" in members
        assert "is_available" in members

    def test_fake_backend_satisfies_protocol(self):
        """FakeBackend is recognised as a BackendProtocol implementation."""
        from constrained_translation.fake_backend import FakeBackend
        from constrained_translation.protocol import BackendProtocol
        fb = FakeBackend(responses={})
        assert isinstance(fb, BackendProtocol)


# ---------------------------------------------------------------------------
# FakeBackend tests
# ---------------------------------------------------------------------------

class TestFakeBackend:
    def _make(self, responses=None):
        from constrained_translation.fake_backend import FakeBackend
        return FakeBackend(responses=responses or {})

    # --- is_available ---

    def test_is_available_returns_true(self):
        fb = self._make()
        assert fb.is_available() is True

    # --- tokenize / decode_token round-trip ---

    def test_tokenize_returns_list_of_ints(self):
        fb = self._make()
        ids = fb.tokenize("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 2  # whitespace split → 2 tokens

    def test_tokenize_single_token(self):
        fb = self._make()
        ids = fb.tokenize("foo")
        assert len(ids) == 1

    def test_tokenize_empty_string(self):
        fb = self._make()
        ids = fb.tokenize("")
        assert ids == []

    def test_decode_token_round_trip(self):
        fb = self._make()
        text = "hello world foo"
        ids = fb.tokenize(text)
        decoded = [fb.decode_token(i) for i in ids]
        assert decoded == text.split()

    def test_same_word_same_id(self):
        """Identical surface forms must map to the same token id."""
        fb = self._make()
        ids1 = fb.tokenize("foo bar foo")
        assert ids1[0] == ids1[2]

    def test_different_words_different_ids(self):
        """Different surface forms must map to different token ids (no collision for simple words)."""
        fb = self._make()
        ids = fb.tokenize("alpha beta gamma")
        assert len(set(ids)) == 3

    def test_decode_unknown_id_raises_or_returns_unk(self):
        """decode_token on an ID not in the vocabulary should raise KeyError or return a fallback string."""
        fb = self._make()
        # Tokenize something to populate the vocab, then try a non-existent id
        fb.tokenize("hello")
        # 99999 is very unlikely to be assigned
        try:
            result = fb.decode_token(99999)
            # If it doesn't raise, it must return a string (UNK fallback)
            assert isinstance(result, str)
        except KeyError:
            pass  # also acceptable

    # --- generate ---

    def test_generate_returns_generation_result(self):
        from constrained_translation.protocol import GenerationResult
        fb = self._make()
        result = fb.generate(prompt="Hello world test", grammar="", max_tokens=10)
        assert isinstance(result, GenerationResult)

    def test_generate_result_has_text(self):
        fb = self._make()
        result = fb.generate(prompt="Hello world test", grammar="", max_tokens=10)
        assert isinstance(result.text, str)

    def test_generate_result_has_token_ids(self):
        fb = self._make()
        result = fb.generate(prompt="Hello world test", grammar="", max_tokens=10)
        assert isinstance(result.token_ids, list)
        assert all(isinstance(i, int) for i in result.token_ids)

    def test_generate_result_token_ids_match_text(self):
        """token_ids in GenerationResult must decode back to the response text."""
        fb = self._make()
        result = fb.generate(prompt="Hello world test", grammar="", max_tokens=10)
        # Each token id should decode to a token present in the response text
        decoded = [fb.decode_token(i) for i in result.token_ids]
        assert decoded == result.text.split()

    def test_generate_uses_canned_response_by_prefix(self):
        """When a 60-char prefix matches, use the canned response."""
        prompt = "A" * 60 + " extra stuff that does not matter"
        canned = "bonjour monde"
        fb = self._make(responses={prompt[:60]: canned})
        result = fb.generate(prompt=prompt, grammar="", max_tokens=20)
        assert result.text == canned

    def test_generate_default_response_when_no_match(self):
        """When no prefix matches, generate returns a non-empty default response."""
        fb = self._make(responses={"XXXXX": "something else"})
        result = fb.generate(prompt="totally different prompt", grammar="", max_tokens=10)
        assert isinstance(result.text, str)
        # Should still be a valid GenerationResult
        assert result.output_tokens >= 0

    def test_generate_prompt_tokens_positive(self):
        fb = self._make()
        result = fb.generate(prompt="some prompt text", grammar="", max_tokens=10)
        assert result.prompt_tokens > 0

    def test_generate_generation_ms_nonnegative(self):
        fb = self._make()
        result = fb.generate(prompt="some prompt text", grammar="", max_tokens=10)
        assert result.generation_ms >= 0.0

    def test_generate_deterministic(self):
        """Same prompt → same token_ids (deterministic hash)."""
        fb = self._make()
        r1 = fb.generate(prompt="stable prompt text here", grammar="g", max_tokens=10)
        r2 = fb.generate(prompt="stable prompt text here", grammar="g", max_tokens=10)
        assert r1.token_ids == r2.token_ids
        assert r1.text == r2.text

    def test_generate_temperature_param_accepted(self):
        """generate() must accept temperature kwarg without raising."""
        fb = self._make()
        result = fb.generate(
            prompt="test temperature param",
            grammar="",
            max_tokens=10,
            temperature=0.7,
        )
        assert isinstance(result.text, str)

    def test_generate_output_tokens_matches_token_ids(self):
        """output_tokens should equal len(token_ids)."""
        fb = self._make()
        result = fb.generate(prompt="check output tokens field", grammar="", max_tokens=10)
        assert result.output_tokens == len(result.token_ids)
