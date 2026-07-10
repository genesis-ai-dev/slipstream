"""BPE / byte-token surface audit regression tests.

These tests expose the blocker that the old _reconstruct_surface_words()
SentencePiece-only approach causes for BPE tokenisers (Qwen3.5, etc.):

Problem A — BPE partial-word splits
  "créa" may tokenise as [cré_id, a_id].  Decoding *individually*:
    decode_token(cré_id) → "cré"
    decode_token(a_id)   → "a"
  The old code sees no ▁ marker, treats each as a separate word,
  checks "cré" and "a" against attested_vocab — both missing → false
  violation even though "créa" IS attested.

Problem B — byte-level BPE tokens (Burmese etc.)
  Individual byte-level tokens can decode to U+FFFD replacement
  characters when decoded in isolation.  The old code passes U+FFFD as
  a "Symbol" layout token, silently skipping the surface word check, but
  the genuine surface word (the reconstructed Burmese character) may never
  be validated at all.

Fix
---
BackendProtocol gains decode_tokens(token_ids: list[int]) -> str which
delegates full-sequence detokenisation to the backend (vLLM /detokenize
with the whole token list).  TokenAuditor.audit() uses decode_tokens for
the surface-composition step and then splits the result with
VocabExtractor semantics (NFKC + Unicode whitespace + punct separation).

Coverage
--------
1.  BackendProtocol exposes decode_tokens().
2.  FakeBackend.decode_tokens() — round-trip.
3.  VLLMBackend.decode_tokens() — exact payload, success, error handling.
4.  BPE créa regression: decode_tokens returns "créa"; audit passes.
5.  BPE créa regression: WITHOUT decode_tokens fix the old path would
    fail; we confirm the new path passes.
6.  Burmese byte-fragment regression: decode_tokens returns proper char.
7.  SpyBackend forwards decode_tokens calls.
8.  SubwordMockBackend.decode_tokens correctly reconstructs from ▁ pieces.
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from constrained_translation.fake_backend import FakeBackend
from constrained_translation.protocol import (
    AlignedExample,
    BackendProtocol,
    GenerationResult,
)
from constrained_translation.token_auditor import TokenAuditor, TokenLicense


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(verse_idx: int, target: str) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source="<src>",
        target=target,
        selection_score=1.0,
        selection_method="semantic",
        evidence_tier=1,
    )


def _make_vllm_response(status_code: int, json_body: object) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = json.dumps(json_body)
    return resp


# ---------------------------------------------------------------------------
# BPE mock backend: individual decode gives partial pieces; full-sequence
# decode gives the correct surface form.
# ---------------------------------------------------------------------------

class _BPEMockBackend:
    """Simulates a Qwen3.5 BPE tokeniser for the créa scenario.

    Token vocabulary:
      300 → decode_token returns "cré"   (first BPE piece of "créa")
      301 → decode_token returns "a"     (second BPE piece of "créa")

    decode_tokens([300, 301]) → "créa"   (proper full-sequence decode)
    """

    _ID2TOK: dict[int, str] = {
        300: "cré",
        301: "a",
    }

    def decode_token(self, token_id: int) -> str:
        return self._ID2TOK[token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Full-sequence detokenise — returns the correct surface form."""
        # Simulates what a real BPE tokeniser's detokenise would return:
        # the pieces joined without any leading space (BPE-style, no ▁).
        return "créa"

    def tokenize(self, text: str) -> list[int]:  # pragma: no cover
        return []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:  # pragma: no cover
        return GenerationResult(
            text="", token_ids=[], prompt_tokens=0, output_tokens=0,
            generation_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Burmese byte-fragment BPE mock backend.
#
# In a real byte-level BPE tokeniser, a single Burmese Unicode character
# (e.g. U+1000 MYANMAR LETTER KA, UTF-8: 0xE1 0x80 0x80) may be split into
# three byte tokens.  Each decoded in isolation yields U+FFFD.
# decode_tokens of the whole sequence returns the real character.
# ---------------------------------------------------------------------------

_BURMESE_KA = "\u1000"  # က — Myanmar Letter Ka


class _BurmeseByteBPEMockBackend:
    """Simulates a byte-level BPE tokeniser splitting a Burmese character.

    Tokens 400, 401, 402 each decode individually to U+FFFD.
    decode_tokens([400, 401, 402]) → the actual Burmese character.
    """

    _ID2TOK: dict[int, str] = {
        400: "\ufffd",
        401: "\ufffd",
        402: "\ufffd",
    }

    def decode_token(self, token_id: int) -> str:
        return self._ID2TOK[token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        return _BURMESE_KA

    def tokenize(self, text: str) -> list[int]:  # pragma: no cover
        return []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:  # pragma: no cover
        return GenerationResult(
            text="", token_ids=[], prompt_tokens=0, output_tokens=0,
            generation_ms=0.0,
        )


# ---------------------------------------------------------------------------
# §1  BackendProtocol exposes decode_tokens
# ---------------------------------------------------------------------------

class TestBackendProtocolDecodesTokens:
    """BackendProtocol must declare decode_tokens(token_ids) -> str."""

    def test_protocol_has_decode_tokens_method(self):
        import inspect
        members = {name for name, _ in inspect.getmembers(BackendProtocol)}
        assert "decode_tokens" in members, (
            "BackendProtocol must expose decode_tokens() for BPE-safe surface audit"
        )

    def test_fake_backend_satisfies_protocol_with_decode_tokens(self):
        """FakeBackend must satisfy BackendProtocol including decode_tokens."""
        fb = FakeBackend()
        assert isinstance(fb, BackendProtocol), (
            "FakeBackend must satisfy BackendProtocol (including decode_tokens)"
        )


# ---------------------------------------------------------------------------
# §2  FakeBackend.decode_tokens — round-trip tests
# ---------------------------------------------------------------------------

class TestFakeBackendDecodeTokens:
    """FakeBackend.decode_tokens() must reconstruct the original text."""

    def test_decode_tokens_method_exists(self):
        fb = FakeBackend()
        assert hasattr(fb, "decode_tokens"), (
            "FakeBackend must implement decode_tokens()"
        )

    def test_decode_tokens_round_trip(self):
        """tokenize then decode_tokens round-trips the original words."""
        fb = FakeBackend()
        ids = fb.tokenize("Au commencement Dieu")
        text = fb.decode_tokens(ids)
        # Round-tripped text must contain all original words
        for word in ["Au", "commencement", "Dieu"]:
            assert word in text, (
                f"decode_tokens result {text!r} missing word {word!r}"
            )

    def test_decode_tokens_empty_list(self):
        fb = FakeBackend()
        assert fb.decode_tokens([]) == ""

    def test_decode_tokens_single_token(self):
        fb = FakeBackend()
        ids = fb.tokenize("bonjour")
        text = fb.decode_tokens(ids)
        assert "bonjour" in text

    def test_decode_tokens_returns_str(self):
        fb = FakeBackend()
        ids = fb.tokenize("hello world")
        result = fb.decode_tokens(ids)
        assert isinstance(result, str)

    def test_decode_tokens_preserves_unicode(self):
        fb = FakeBackend()
        ids = fb.tokenize("créa lumière")
        text = fb.decode_tokens(ids)
        assert "créa" in text
        assert "lumière" in text


# ---------------------------------------------------------------------------
# §3  VLLMBackend.decode_tokens — exact payload and success path
# ---------------------------------------------------------------------------

class TestVLLMBackendDecodeTokens:
    """VLLMBackend.decode_tokens() must POST the full token list to /detokenize."""

    def _make_backend(self):
        from constrained_translation.vllm_backend import VLLMBackend
        return VLLMBackend(base_url="http://localhost:8000", model="test-model")

    def test_decode_tokens_method_exists(self):
        b = self._make_backend()
        assert hasattr(b, "decode_tokens"), (
            "VLLMBackend must implement decode_tokens()"
        )

    def test_decode_tokens_posts_to_detokenize_endpoint(self):
        b = self._make_backend()
        resp = _make_vllm_response(200, {"prompt": "Au commencement"})
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_tokens([101, 202])
        url = mock_post.call_args[0][0]
        assert url.endswith("/detokenize"), (
            f"decode_tokens must POST to /detokenize, got URL: {url}"
        )

    def test_decode_tokens_payload_contains_all_token_ids(self):
        """The full token list is sent in one request (not one per ID)."""
        b = self._make_backend()
        resp = _make_vllm_response(200, {"prompt": "créa"})
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_tokens([300, 301])
        body = mock_post.call_args[1]["json"]
        assert body["tokens"] == [300, 301], (
            "decode_tokens must send all token IDs in a single request"
        )

    def test_decode_tokens_payload_contains_model(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="qwen-model")
        resp = _make_vllm_response(200, {"prompt": "hello"})
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_tokens([1])
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "qwen-model"

    def test_decode_tokens_returns_prompt_field(self):
        b = self._make_backend()
        resp = _make_vllm_response(200, {"prompt": "Au commencement"})
        with patch("requests.post", return_value=resp):
            result = b.decode_tokens([101, 202])
        assert result == "Au commencement"

    def test_decode_tokens_empty_list_returns_empty_string(self):
        """Empty token list must return '' without making an HTTP request."""
        b = self._make_backend()
        with patch("requests.post") as mock_post:
            result = b.decode_tokens([])
        assert result == ""
        mock_post.assert_not_called()

    def test_decode_tokens_missing_prompt_key_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = self._make_backend()
        resp = _make_vllm_response(200, {"tokens": [1, 2]})  # wrong key
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.decode_tokens([1, 2])
        assert exc_info.value.operation == "decode_tokens"

    def test_decode_tokens_http_error_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = self._make_backend()
        resp = _make_vllm_response(500, {"error": "server error"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.decode_tokens([1, 2])
        assert exc_info.value.status == 500

    def test_decode_tokens_single_call_for_whole_sequence(self):
        """decode_tokens must make exactly ONE HTTP call (not one per ID)."""
        b = self._make_backend()
        resp = _make_vllm_response(200, {"prompt": "hello world foo"})
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_tokens([1, 2, 3])
        assert mock_post.call_count == 1, (
            "decode_tokens must issue exactly one /detokenize request "
            "regardless of sequence length"
        )

    def test_decode_tokens_unicode_preserved(self):
        """Unicode output from /detokenize is returned verbatim."""
        b = self._make_backend()
        resp = _make_vllm_response(200, {"prompt": "créa"})
        with patch("requests.post", return_value=resp):
            result = b.decode_tokens([300, 301])
        assert result == "créa"


# ---------------------------------------------------------------------------
# §4  BPE créa regression: audit must PASS when decode_tokens is used
# ---------------------------------------------------------------------------

class TestBPECreaSurfaceAudit:
    """Regression: BPE 'créa' split into two pieces must not cause false audit failure.

    Old (broken) behaviour:
      decode_token(300) = "cré", decode_token(301) = "a"
      _reconstruct_surface_words: no ▁ marker → each piece is its own word
      Surface check: "cré" not in attested_vocab, "a" not in attested_vocab
      → FALSE violation even though "créa" IS attested

    New (correct) behaviour:
      decode_tokens([300, 301]) = "créa"
      Split "créa" → ["créa"]
      "créa" IS in attested_vocab → PASS
    """

    def test_bpe_crea_pieces_pass_with_decode_tokens(self):
        """BPE 'créa' split into ['cré', 'a'] must pass when attested_vocab has 'créa'."""
        backend = _BPEMockBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {300: [0], 301: [0]}
        attested_vocab = frozenset({"créa"})

        result, _ = auditor.audit(
            token_ids=[300, 301],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True, (
            f"BPE 'créa' split into ['cré','a'] must PASS when 'créa' is in "
            f"attested_vocab.  Violations: {result.violations}"
        )
        assert result.violations == [], (
            f"No violations expected for BPE créa; got: {result.violations}"
        )

    def test_bpe_crea_still_fails_when_not_in_vocab(self):
        """If 'créa' is NOT in attested_vocab the audit must still fail."""
        backend = _BPEMockBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {300: [0], 301: [0]}
        attested_vocab = frozenset({"dieu", "monde"})  # "créa" NOT here

        result, _ = auditor.audit(
            token_ids=[300, 301],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "BPE 'créa' must FAIL when 'créa' is absent from attested_vocab"
        )
        assert any("créa" in v for v in result.violations), (
            f"Expected violation mentioning 'créa'; got: {result.violations}"
        )

    def test_bpe_decode_tokens_is_called_for_surface_check(self):
        """audit() must call decode_tokens() for the surface-composition step."""

        calls: list[list[int]] = []

        class _TrackingBPEBackend(_BPEMockBackend):
            def decode_tokens(self, token_ids: list[int]) -> str:
                calls.append(list(token_ids))
                return super().decode_tokens(token_ids)

        backend = _TrackingBPEBackend()
        auditor = TokenAuditor()
        licence: TokenLicense = {300: [0], 301: [0]}

        auditor.audit(
            token_ids=[300, 301],
            attested_vocab=frozenset({"créa"}),
            backend=backend,
            license=licence,
        )

        assert calls, "decode_tokens must be called during audit() for surface check"
        assert [300, 301] in calls, (
            f"decode_tokens must be called with [300, 301]; calls: {calls}"
        )


# ---------------------------------------------------------------------------
# §5  Burmese byte-fragment regression
# ---------------------------------------------------------------------------

class TestBurmeseByteFragmentAudit:
    """Regression: byte-level BPE Burmese fragments must not cause false failures.

    Individual byte tokens decode to U+FFFD in isolation.
    decode_tokens of the full sequence returns the actual Burmese character.
    """

    def test_burmese_byte_fragments_pass_with_decode_tokens(self):
        """Three U+FFFD byte tokens that form a Burmese char must pass audit."""
        backend = _BurmeseByteBPEMockBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {400: [0], 401: [0], 402: [0]}
        attested_vocab = frozenset({_BURMESE_KA})  # the reconstructed char

        result, _ = auditor.audit(
            token_ids=[400, 401, 402],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True, (
            f"Burmese byte fragments must PASS when the reconstructed character "
            f"is in attested_vocab.  Violations: {result.violations}"
        )

    def test_burmese_byte_fragments_fail_when_char_not_in_vocab(self):
        """Burmese char must fail if NOT in attested_vocab."""
        backend = _BurmeseByteBPEMockBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {400: [0], 401: [0], 402: [0]}
        attested_vocab = frozenset({"other_word"})  # Myanmar Ka NOT here

        result, _ = auditor.audit(
            token_ids=[400, 401, 402],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "Burmese byte fragments must FAIL when the Burmese char is absent"
        )


# ---------------------------------------------------------------------------
# §6  Existing SentencePiece scenario still works with decode_tokens
# ---------------------------------------------------------------------------

class TestSentencePieceStillWorks:
    """Verify existing SentencePiece ▁ test scenarios work via decode_tokens."""

    class _SPBackend:
        """SentencePiece backend that properly implements decode_tokens."""

        _ID2TOK: dict[int, str] = {
            100: "▁hel",
            101: "lo",
            102: "▁foo",
            103: "▁bar",
        }

        def decode_token(self, token_id: int) -> str:
            return self._ID2TOK[token_id]

        def decode_tokens(self, token_ids: list[int]) -> str:
            """Reconstruct full text from SentencePiece pieces."""
            pieces = [self._ID2TOK[tid] for tid in token_ids]
            words: list[str] = []
            current = ""
            for piece in pieces:
                if piece.startswith("▁"):
                    if current:
                        words.append(current)
                    current = piece[1:]
                else:
                    current += piece
            if current:
                words.append(current)
            return " ".join(words)

        def tokenize(self, text: str) -> list[int]:  # pragma: no cover
            return []

        def is_available(self) -> bool:
            return True

        def generate(self, *args, **kwargs) -> GenerationResult:  # pragma: no cover
            return GenerationResult(
                text="", token_ids=[], prompt_tokens=0, output_tokens=0,
                generation_ms=0.0,
            )

    def test_sp_decode_tokens_reconstructs_hello(self):
        """▁hel + lo → decode_tokens returns 'hello'."""
        backend = self._SPBackend()
        assert backend.decode_tokens([100, 101]) == "hello"

    def test_sp_recombined_unattested_still_fails(self):
        """SentencePiece recombination test still catches unattested words."""
        backend = self._SPBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {100: [0], 101: [1]}
        attested_vocab = frozenset({"hel", "lo"})  # NOT "hello"

        result, _ = auditor.audit(
            token_ids=[100, 101],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "Recombined SentencePiece pieces forming 'hello' must still fail "
            "when 'hello' is not in attested_vocab."
        )
        assert any("hello" in v for v in result.violations), (
            f"Violation must mention 'hello'; got: {result.violations}"
        )

    def test_sp_recombined_attested_passes(self):
        """SentencePiece pieces forming a word that IS attested must pass."""
        backend = self._SPBackend()
        auditor = TokenAuditor()

        licence: TokenLicense = {100: [0], 101: [0]}
        attested_vocab = frozenset({"hello"})  # "hello" IS attested

        result, _ = auditor.audit(
            token_ids=[100, 101],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True, (
            f"SP pieces forming attested 'hello' must pass; "
            f"violations: {result.violations}"
        )


# ---------------------------------------------------------------------------
# §7  FakeBackend with TokenAuditor — end-to-end surface audit path
# ---------------------------------------------------------------------------

class TestFakeBackendEndToEnd:
    """FakeBackend.decode_tokens integrates correctly with TokenAuditor."""

    def test_fake_backend_audit_passes_with_attested_words(self):
        """Full audit pass using FakeBackend and decode_tokens path."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="Au commencement Dieu créa")
        token_ids = fb.tokenize("Au commencement")
        attested_vocab = frozenset({"Au", "commencement", "Dieu", "créa"})
        auditor = TokenAuditor()

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True, (
            f"FakeBackend end-to-end audit must pass; "
            f"violations: {result.violations}"
        )

    def test_decode_tokens_called_for_surface_composition(self):
        """audit() calls decode_tokens for the surface-composition step."""
        fb = FakeBackend()
        calls: list = []

        class _SpyFakeBackend(FakeBackend):
            def decode_tokens(self, token_ids: list[int]) -> str:
                calls.append(list(token_ids))
                return super().decode_tokens(token_ids)

        spy_fb = _SpyFakeBackend()
        ex = _make_example(verse_idx=0, target="Au commencement")
        token_ids = spy_fb.tokenize("Au commencement")
        attested_vocab = frozenset({"Au", "commencement"})
        auditor = TokenAuditor()

        auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=spy_fb,
            examples=[ex],
        )

        assert calls, "audit() must call decode_tokens() for surface composition"
