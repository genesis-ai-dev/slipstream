"""Tests for constrained_translation.token_auditor — Task 8.

Strict TDD: all tests are written *before* the implementation.
Run them first to confirm RED, then implement token_auditor.py to go GREEN.

Coverage invariants exercised
──────────────────────────────
1. Exact allowed IDs pass (all IDs licensed from examples).
2. A foreign (unlicensed) ID fails.
3. Two individually licensed subword pieces recombined into an unattested
   surface word fail (surface-composition check).
4. Model-generated [UNK:…] tokens always fail (invariant I3).
5. Punctuation tokens pass under the documented narrow rule.
6. Whitespace-only tokens pass as layout.
7. Violations list is populated with readable strings on failure.
8. backend.decode_token and backend.tokenize are actually called.
9. Every passing non-layout token ID maps to at least one verse_idx in the
   returned provenance map.
10. Precomputed licence is accepted in lieu of examples.
11. build_license maps each token ID to the correct verse_idx list.
"""
from __future__ import annotations

import pytest

from constrained_translation.fake_backend import FakeBackend
from constrained_translation.protocol import (
    AlignedExample,
    GenerationResult,
    TokenAuditResult,
)

# ── Import under test ───────────────────────────────────────────────────────
# These will fail (ImportError) until token_auditor.py is created.
from constrained_translation.token_auditor import TokenAuditor, TokenLicense


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers / fixtures
# ════════════════════════════════════════════════════════════════════════════

def _make_example(verse_idx: int, target: str) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source="<src>",
        target=target,
        selection_score=1.0,
        selection_method="semantic",
        evidence_tier=1,
    )


# ---------------------------------------------------------------------------
# Minimal subword-style backend for the surface-composition recombination test.
#
# Tokens without a leading ▁ (U+2581) are treated as continuation pieces by
# the auditor's _reconstruct_surface_words() helper.
# ---------------------------------------------------------------------------

class _SubwordMockBackend:
    """Simulates a SentencePiece-style subword tokenizer (no actual model).

    Vocabulary (hardcoded for test isolation):
      ▁hel → 100  (word-start piece, decodes with ▁ prefix)
      lo   → 101  (continuation piece, no word-boundary marker)
      ▁foo → 102
      ▁bar → 103
    """

    _ID2TOK: dict[int, str] = {
        100: "▁hel",
        101: "lo",
        102: "▁foo",
        103: "▁bar",
    }
    _TOK2ID: dict[str, int] = {v: k for k, v in _ID2TOK.items()}

    def decode_token(self, token_id: int) -> str:
        return self._ID2TOK[token_id]

    def tokenize(self, text: str) -> list[int]:
        """Return IDs for tokens whose ▁-prefixed form is in the vocabulary."""
        result: list[int] = []
        for word in text.split():
            tok = "\u2581" + word  # ▁ + word
            if tok in self._TOK2ID:
                result.append(self._TOK2ID[tok])
            # continuation pieces are never emitted by tokenize() for full text
        return result

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
            text="",
            token_ids=[],
            prompt_tokens=0,
            output_tokens=0,
            generation_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Spy wrapper — records calls to decode_token / tokenize
# ---------------------------------------------------------------------------

class _SpyBackend:
    """Wraps a BackendProtocol implementation and records method calls."""

    def __init__(self, inner: FakeBackend) -> None:
        self._inner = inner
        self.decode_calls: list[int] = []
        self.tokenize_calls: list[str] = []

    def decode_token(self, token_id: int) -> str:
        self.decode_calls.append(token_id)
        return self._inner.decode_token(token_id)

    def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls.append(text)
        return self._inner.tokenize(text)

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:  # pragma: no cover
        return self._inner.generate(prompt, grammar, max_tokens, temperature)


# ════════════════════════════════════════════════════════════════════════════
# Test suite
# ════════════════════════════════════════════════════════════════════════════


class TestBuildLicense:
    """TokenAuditor.build_license() builds a correct token-ID → verse_idx map."""

    def test_build_license_maps_ids_to_verse_idx(self):
        """Each token ID from backend.tokenize(example.target) maps to verse_idx."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=3, target="Au commencement")
        auditor = TokenAuditor()

        licence = auditor.build_license([ex], fb)

        # All token IDs that FakeBackend assigns to "Au commencement" must be present.
        expected_ids = fb.tokenize("Au commencement")
        assert expected_ids, "FakeBackend must produce IDs for non-empty text"
        for tid in expected_ids:
            assert tid in licence
            assert 3 in licence[tid], "verse_idx=3 must be in the provenance list"

    def test_build_license_multiple_examples_accumulate_verse_idx(self):
        """An ID appearing in two examples lists both verse_idx values."""
        fb = FakeBackend()
        # Both examples share the token "Dieu".
        ex0 = _make_example(verse_idx=0, target="Dieu créa")
        ex1 = _make_example(verse_idx=1, target="Dieu vit")
        auditor = TokenAuditor()

        licence = auditor.build_license([ex0, ex1], fb)

        dieu_id = fb.tokenize("Dieu")[0]
        assert dieu_id in licence
        assert 0 in licence[dieu_id]
        assert 1 in licence[dieu_id]

    def test_build_license_empty_examples_returns_empty(self):
        fb = FakeBackend()
        auditor = TokenAuditor()
        assert auditor.build_license([], fb) == {}

    def test_build_license_uses_backend_tokenize(self):
        """build_license calls backend.tokenize for every example.target."""
        fb = FakeBackend()
        spy = _SpyBackend(fb)
        ex0 = _make_example(verse_idx=0, target="Au commencement")
        ex1 = _make_example(verse_idx=1, target="Dieu créa")
        auditor = TokenAuditor()

        auditor.build_license([ex0, ex1], spy)

        assert "Au commencement" in spy.tokenize_calls
        assert "Dieu créa" in spy.tokenize_calls


# ════════════════════════════════════════════════════════════════════════════


class TestAuditPassCases:
    """Passing scenarios — all valid, licensed tokens."""

    # ── Test 1 ──────────────────────────────────────────────────────────────

    def test_auditor_pass_when_all_tokens_attested(self):
        """All generated IDs licensed from examples → audit passes."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="Au commencement Dieu créa")
        token_ids = fb.tokenize("Au commencement")
        attested_vocab = frozenset({"Au", "commencement", "Dieu", "créa"})
        auditor = TokenAuditor()

        result, provenance = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert isinstance(result, TokenAuditResult)
        assert result.passed is True
        assert result.violations == []

    def test_auditor_pass_returns_tuple_of_result_and_provenance(self):
        """audit() returns a 2-tuple: (TokenAuditResult, TokenLicense dict)."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="bonjour")
        ids = fb.tokenize("bonjour")
        auditor = TokenAuditor()

        result, provenance = auditor.audit(
            token_ids=ids,
            attested_vocab=frozenset({"bonjour"}),
            backend=fb,
            examples=[ex],
        )

        assert isinstance(result, TokenAuditResult)
        assert isinstance(provenance, dict)


# ════════════════════════════════════════════════════════════════════════════


class TestAuditFailCases:
    """Failure scenarios — foreign IDs, UNK markers, recombination."""

    # ── Test 2: Foreign ID ──────────────────────────────────────────────────

    def test_auditor_fail_when_foreign_token_present(self):
        """An ID not derived from any selected example is a provenance violation."""
        fb = FakeBackend()
        # License is built from "bonjour" only.
        ex = _make_example(verse_idx=0, target="bonjour")
        auditor = TokenAuditor()

        # Generate "foreign" — never appears in any example target.
        foreign_id = fb.tokenize("foreign")[0]
        attested_vocab = frozenset({"bonjour"})  # "foreign" is NOT here either

        result, _ = auditor.audit(
            token_ids=[foreign_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is False
        # Violation message must mention the offending token ID.
        assert any(str(foreign_id) in v for v in result.violations)

    # ── Test 3: Surface-composition / recombination ─────────────────────────

    def test_auditor_recombined_pieces_not_in_attested_fail(self):
        """Two individually-licensed subword pieces that reconstruct an unattested
        surface word are a surface-composition violation.

        Scenario
        --------
        • Token 100 decodes to '▁hel'  (licensed from verse_idx=0)
        • Token 101 decodes to 'lo'    (licensed from verse_idx=1,
                                        no ▁ marker → continuation of prev)
        • Reconstructed word: 'hello'
        • attested_vocab = {'hel', 'lo'}  (does NOT contain 'hello')
        • Expected: violation for surface word 'hello'
        """
        backend = _SubwordMockBackend()
        auditor = TokenAuditor()

        # Precomputed licence: both IDs are licensed but from different verses.
        precomputed_licence: TokenLicense = {100: [0], 101: [1]}
        attested_vocab = frozenset({"hel", "lo"})  # NOT "hello"

        result, _ = auditor.audit(
            token_ids=[100, 101],
            attested_vocab=attested_vocab,
            backend=backend,
            license=precomputed_licence,
        )

        assert result.passed is False, (
            "Recombined subword pieces forming an unattested word must fail."
        )
        assert any("hello" in v for v in result.violations), (
            f"Expected 'hello' in a violation string; got: {result.violations}"
        )

    # ── Test 4: Model-generated UNK markers ─────────────────────────────────

    def test_auditor_unk_markers_fail_in_model_output(self):
        """[UNK:…] appearing in model output is always a violation (invariant I3).

        UNK markers are inserted deterministically by UNKDetector in the source
        text *before* generation.  The model must never emit them spontaneously.
        """
        fb = FakeBackend()
        auditor = TokenAuditor()

        # Build a licence that does NOT contain the UNK token.
        ex = _make_example(verse_idx=0, target="Au commencement")
        unk_id = fb.tokenize("[UNK:selah]")[0]
        attested_vocab = frozenset({"Au", "commencement"})

        result, _ = auditor.audit(
            token_ids=[unk_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is False
        assert any("UNK" in v for v in result.violations), (
            f"Expected a UNK-marker violation; got: {result.violations}"
        )

    def test_auditor_unk_marker_fails_even_if_in_licence(self):
        """[UNK:…] fails even if the token ID somehow ended up in the licence."""
        fb = FakeBackend()
        auditor = TokenAuditor()

        unk_text = "[UNK:selah]"
        unk_id = fb.tokenize(unk_text)[0]
        # Force the UNK token into the licence via a specially crafted example.
        ex = _make_example(verse_idx=0, target=unk_text)
        attested_vocab = frozenset({unk_text})

        result, _ = auditor.audit(
            token_ids=[unk_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is False
        assert any("UNK" in v for v in result.violations)


# ════════════════════════════════════════════════════════════════════════════


class TestLayoutPolicy:
    """Punctuation and whitespace narrow-rule tests."""

    # ── Test 5: Punctuation ──────────────────────────────────────────────────

    def test_auditor_punctuation_passes(self):
        """Tokens that decode to pure punctuation pass under the layout rule.

        Documented narrow rule: punctuation characters are structural tokens,
        not lexical content.  They pass regardless of licence membership.
        """
        fb = FakeBackend()
        auditor = TokenAuditor()

        # Punctuation tokens will NOT appear in any example target → not licenced.
        punct_id = fb.tokenize(".")[0]
        attested_vocab = frozenset({"Au", "commencement"})
        # Empty examples → licence contains only example targets, which have no ".".
        ex = _make_example(verse_idx=0, target="Au commencement")

        result, _ = auditor.audit(
            token_ids=[punct_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True, (
            f"Pure punctuation token should pass; violations: {result.violations}"
        )

    def test_auditor_multiple_punctuation_tokens_pass(self):
        """Comma, exclamation, colon all pass under the layout narrow rule."""
        fb = FakeBackend()
        auditor = TokenAuditor()
        ex = _make_example(verse_idx=0, target="Au commencement")
        attested_vocab = frozenset({"Au", "commencement"})

        for punct in (",", "!", ":"):
            punct_id = fb.tokenize(punct)[0]
            result, _ = auditor.audit(
                token_ids=[punct_id],
                attested_vocab=attested_vocab,
                backend=fb,
                examples=[ex],
            )
            assert result.passed is True, (
                f"Punctuation {punct!r} should pass; violations: {result.violations}"
            )

    # ── Test 6: Whitespace ───────────────────────────────────────────────────

    def test_auditor_whitespace_passes(self):
        """Whitespace-only tokens pass unconditionally as layout tokens.

        The FakeBackend splits on whitespace so pure-whitespace tokens would
        never normally appear; we inject one via a precomputed licence to
        simulate a real tokenizer that emits standalone space tokens.
        """
        fb = FakeBackend()
        auditor = TokenAuditor()

        # We need a backend that can decode a whitespace-only token.
        # Inject the token directly into FakeBackend's vocab.
        space_id = fb._get_or_assign_id(" ")  # internal helper for test setup
        attested_vocab = frozenset({"Au"})
        ex = _make_example(verse_idx=0, target="Au")

        result, _ = auditor.audit(
            token_ids=[space_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True, (
            f"Whitespace-only token should pass; violations: {result.violations}"
        )

    def test_auditor_whitespace_not_in_provenance_map(self):
        """Whitespace-only tokens are layout and do not appear in provenance_map."""
        fb = FakeBackend()
        auditor = TokenAuditor()
        space_id = fb._get_or_assign_id(" ")
        ex = _make_example(verse_idx=0, target="Au")
        attested_vocab = frozenset({"Au"})

        _, provenance = auditor.audit(
            token_ids=[space_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        # Whitespace tokens are layout; they carry no lexical provenance.
        assert space_id not in provenance


# ════════════════════════════════════════════════════════════════════════════


class TestViolationsAndProvenance:
    """Violations list population and provenance map correctness."""

    # ── Test 7: Violations list ──────────────────────────────────────────────

    def test_auditor_violations_list_is_populated(self):
        """On failure, violations contains at least one non-empty string per bad token."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="bonjour")
        auditor = TokenAuditor()

        bad_id = fb.tokenize("unattested_word")[0]
        attested_vocab = frozenset({"bonjour"})

        result, _ = auditor.audit(
            token_ids=[bad_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert not result.passed
        assert len(result.violations) >= 1
        for v in result.violations:
            assert isinstance(v, str)
            assert v  # non-empty

    def test_auditor_violation_mentions_token_id(self):
        """A licence-violation string must contain the offending token_id."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="bonjour")
        bad_id = fb.tokenize("xyz_bad")[0]
        attested_vocab = frozenset({"bonjour"})
        auditor = TokenAuditor()

        result, _ = auditor.audit(
            token_ids=[bad_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert any(str(bad_id) in v for v in result.violations)

    def test_auditor_violation_mentions_decoded_surface(self):
        """A licence-violation string must include the decoded surface form."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="bonjour")
        bad_id = fb.tokenize("surface_xyz")[0]
        attested_vocab = frozenset({"bonjour"})
        auditor = TokenAuditor()

        result, _ = auditor.audit(
            token_ids=[bad_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert any("surface_xyz" in v for v in result.violations)

    # ── Test 9: Provenance map ───────────────────────────────────────────────

    def test_every_passing_non_layout_id_maps_to_verse_idx(self):
        """Every passing, non-layout token ID has a verse_idx entry in provenance_map."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=7, target="Au commencement Dieu")
        token_ids = fb.tokenize("Au commencement Dieu")
        attested_vocab = frozenset({"Au", "commencement", "Dieu"})
        auditor = TokenAuditor()

        result, provenance = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True
        for tid in token_ids:
            decoded = fb.decode_token(tid)
            if decoded.strip():  # skip whitespace-only (layout)
                assert tid in provenance, (
                    f"token_id={tid} (decoded={decoded!r}) missing from provenance_map"
                )
                assert 7 in provenance[tid], (
                    f"verse_idx=7 missing from provenance_map[{tid}]"
                )

    def test_provenance_map_lists_all_verse_indices(self):
        """A token appearing in two examples lists both verse_idx values."""
        fb = FakeBackend()
        ex0 = _make_example(verse_idx=0, target="Dieu créa")
        ex1 = _make_example(verse_idx=1, target="Dieu vit")
        dieu_id = fb.tokenize("Dieu")[0]
        attested_vocab = frozenset({"Dieu", "créa", "vit"})
        auditor = TokenAuditor()

        result, provenance = auditor.audit(
            token_ids=[dieu_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex0, ex1],
        )

        assert result.passed is True
        assert dieu_id in provenance
        assert 0 in provenance[dieu_id]
        assert 1 in provenance[dieu_id]


# ════════════════════════════════════════════════════════════════════════════


class TestBackendUsage:
    """Test 8 — backend.decode_token and backend.tokenize are actually called."""

    def test_auditor_uses_fake_backend_decode(self):
        """audit() calls backend.decode_token for every token_id in token_ids."""
        fb = FakeBackend()
        spy = _SpyBackend(fb)
        ex = _make_example(verse_idx=0, target="Au commencement")
        token_ids = fb.tokenize("Au")
        attested_vocab = frozenset({"Au", "commencement"})
        auditor = TokenAuditor()

        auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=spy,
            examples=[ex],
        )

        assert spy.decode_calls, "backend.decode_token must be called during audit()"
        for tid in token_ids:
            assert tid in spy.decode_calls

    def test_auditor_uses_backend_tokenize_for_licence(self):
        """When examples are provided, backend.tokenize is called to build licence."""
        fb = FakeBackend()
        spy = _SpyBackend(fb)
        ex = _make_example(verse_idx=0, target="Au commencement")
        token_ids = fb.tokenize("Au")
        attested_vocab = frozenset({"Au", "commencement"})
        auditor = TokenAuditor()

        auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=spy,
            examples=[ex],
        )

        assert spy.tokenize_calls, "backend.tokenize must be called to build the licence"
        assert "Au commencement" in spy.tokenize_calls


# ════════════════════════════════════════════════════════════════════════════


class TestPrecomputedLicence:
    """Test 10 — precomputed licence bypasses build_license."""

    def test_auditor_accepts_precomputed_license(self):
        """Passing a precomputed licence skips build_license; no extra tokenize calls."""
        fb = FakeBackend()
        spy = _SpyBackend(fb)
        token_id = fb.tokenize("bonjour")[0]
        precomputed: TokenLicense = {token_id: [42]}
        attested_vocab = frozenset({"bonjour"})
        auditor = TokenAuditor()

        result, provenance = auditor.audit(
            token_ids=[token_id],
            attested_vocab=attested_vocab,
            backend=spy,
            license=precomputed,
        )

        assert result.passed is True
        # build_license was not called, so tokenize should NOT have been called.
        assert spy.tokenize_calls == [], (
            "tokenize should not be called when precomputed licence is supplied"
        )
        assert token_id in provenance
        assert 42 in provenance[token_id]

    def test_auditor_raises_without_examples_and_license(self):
        """ValueError raised if neither examples nor license is supplied."""
        fb = FakeBackend()
        auditor = TokenAuditor()

        with pytest.raises(ValueError, match="examples.*license|license.*examples"):
            auditor.audit(
                token_ids=[0],
                attested_vocab=frozenset(),
                backend=fb,
            )


# ════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Empty inputs, duplicate IDs, and mixed passing/failing tokens."""

    def test_empty_token_ids_passes(self):
        """An empty token_ids list always passes with empty violations."""
        fb = FakeBackend()
        auditor = TokenAuditor()
        ex = _make_example(verse_idx=0, target="Au")

        result, provenance = auditor.audit(
            token_ids=[],
            attested_vocab=frozenset({"Au"}),
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True
        assert result.violations == []
        assert provenance == {}

    def test_mixed_tokens_one_bad_fails_overall(self):
        """If any token fails, the entire audit fails."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="Au")
        good_id = fb.tokenize("Au")[0]
        bad_id = fb.tokenize("unattested")[0]
        attested_vocab = frozenset({"Au"})
        auditor = TokenAuditor()

        result, _ = auditor.audit(
            token_ids=[good_id, bad_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is False

    def test_duplicate_ids_handled_gracefully(self):
        """Duplicate token_ids do not raise; each occurrence is individually audited."""
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="Au commencement")
        au_id = fb.tokenize("Au")[0]
        attested_vocab = frozenset({"Au", "commencement"})
        auditor = TokenAuditor()

        # Repeat the same ID three times.
        result, provenance = auditor.audit(
            token_ids=[au_id, au_id, au_id],
            attested_vocab=attested_vocab,
            backend=fb,
            examples=[ex],
        )

        assert result.passed is True
        assert au_id in provenance


# ════════════════════════════════════════════════════════════════════════════
# Quality-review tests: decode caching, surface-composition only for licensed
# ════════════════════════════════════════════════════════════════════════════


class TestDecodeCaching:
    """decode_token results must be cached by unique token ID (called once per ID)."""

    def test_decode_called_once_per_unique_id(self):
        """backend.decode_token must be called exactly once per unique token ID,
        even when the same ID appears multiple times in token_ids.
        """
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="Au commencement Dieu")
        au_id = fb.tokenize("Au")[0]
        comm_id = fb.tokenize("commencement")[0]
        attested_vocab = frozenset({"Au", "commencement", "Dieu"})

        decode_call_counts: dict = {}

        class _CountingBackend:
            def decode_token(self, token_id: int) -> str:
                decode_call_counts[token_id] = decode_call_counts.get(token_id, 0) + 1
                return fb.decode_token(token_id)

            def tokenize(self, text: str) -> list:
                return fb.tokenize(text)

            def is_available(self) -> bool:
                return True

            def generate(self, *args, **kwargs):  # pragma: no cover
                return fb.generate(*args, **kwargs)

        auditor = TokenAuditor()
        # au_id appears 3 times, comm_id appears 2 times.
        auditor.audit(
            token_ids=[au_id, comm_id, au_id, au_id, comm_id],
            attested_vocab=attested_vocab,
            backend=_CountingBackend(),
            examples=[ex],
        )

        # Each unique ID must have been decoded exactly once.
        assert decode_call_counts.get(au_id, 0) == 1, (
            f"decode_token called {decode_call_counts.get(au_id, 0)} times for "
            f"au_id={au_id}; must be called exactly once per unique ID"
        )
        assert decode_call_counts.get(comm_id, 0) == 1, (
            f"decode_token called {decode_call_counts.get(comm_id, 0)} times for "
            f"comm_id={comm_id}; must be called exactly once per unique ID"
        )


class TestSurfaceCompositionOnlyForLicensedIDs:
    """Surface-composition violations must not be generated for unlicensed IDs.

    An unlicensed ID already produces an ID-level licence violation at step 2c.
    Adding a surface-composition violation for the same token would double-count
    the failure and obscure the actual provenance problem.
    """

    def test_unlicensed_id_produces_id_violation_not_surface_violation(self):
        """When a token ID is not in the licence, the violation is an ID-level
        licence failure, not a surface-composition failure.

        Setup:
          - Example target: "hello world"
          - Licence: contains IDs for "hello" and "world"
          - Generated token: ID for "foreign" (not in licence, not in attested_vocab)
          - attested_vocab: {"hello", "world"}

        Expected: one violation mentioning token_id, no surface violation.
        """
        fb = FakeBackend()
        ex = _make_example(verse_idx=0, target="hello world")
        # Build licence: IDs for hello + world.
        auditor = TokenAuditor()
        licence = auditor.build_license([ex], fb)

        # Now generate a token that is NOT in the licence.
        foreign_id = fb.tokenize("foreign")[0]
        assert foreign_id not in licence, "test pre-condition: foreign must not be licensed"

        attested_vocab = frozenset({"hello", "world"})

        result, _ = auditor.audit(
            token_ids=[foreign_id],
            attested_vocab=attested_vocab,
            backend=fb,
            license=licence,
        )

        assert not result.passed
        # There must be exactly one violation (ID-level), not two (ID + surface).
        # The violation must mention the token_id.
        id_violations = [v for v in result.violations if str(foreign_id) in v]
        surface_violations = [v for v in result.violations if "subword recombination" in v]
        assert len(id_violations) == 1, (
            f"Expected exactly 1 ID-level violation for foreign_id={foreign_id}; "
            f"got: {result.violations}"
        )
        assert len(surface_violations) == 0, (
            f"Surface-composition check must NOT fire for unlicensed IDs; "
            f"got surface violations: {surface_violations}"
        )

    def test_licensed_ids_still_checked_for_surface_composition(self):
        """Even when all IDs are licensed, novel recombined surface words still fail.

        Two pieces are individually licensed (from different verses) but their
        reconstructed surface word is not in attested_vocab.
        """
        backend = _SubwordMockBackend()
        auditor = TokenAuditor()

        # Both IDs are licensed.
        precomputed_licence: TokenLicense = {100: [0], 101: [1]}
        # "hello" (reconstructed from ▁hel + lo) is NOT in attested_vocab.
        attested_vocab = frozenset({"hel", "lo"})

        result, _ = auditor.audit(
            token_ids=[100, 101],
            attested_vocab=attested_vocab,
            backend=backend,
            license=precomputed_licence,
        )

        assert not result.passed, (
            "Recombined surface word 'hello' is not in attested_vocab → must fail"
        )
        surface_violations = [v for v in result.violations if "hello" in v]
        assert surface_violations, (
            f"Expected a surface violation mentioning 'hello'; got: {result.violations}"
        )
