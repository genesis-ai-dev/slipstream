"""Tests for Qwen im_end terminal control ID handling in TokenAuditor.

Context
-------
Qwen/vLLM returns a terminal control token (ID 248046, surface '<|im_end|>')
in GenerationResult.token_ids even though GenerationResult.text omits it.
This caused 152/360 semantic+coverage audit records to fail solely because of
the appended protocol control marker.

Why the backend terminal ID differs from emitted text
------------------------------------------------------
vLLM returns the raw token stream including the model's own end-of-generation
control token (im_end).  The model emits it to signal stop; vLLM strips it
from the displayed text but preserves it in token_ids for audit purposes.
The token is a *protocol* artefact, never a lexical output of the model, so
it must not be treated as lexical content by the provenance auditor.

Required behaviour (TDD — RED first, then implementation)
-----------------------------------------------------------
1.  Exactly '<|im_end|>' is the recognised terminal control surface (and
    only '<|im_end|>' — not any generic angle-bracket string).
2.  Final-position-only: '<|im_end|>' is silently skipped when it is the
    *last* token in the sequence.
3.  At any non-final position '<|im_end|>' is a *loud violation*.
4.  Multiple '<|im_end|>' tokens: the non-final occurrence(s) fail loudly;
    only one may appear at the very end.
5.  Control-only sequence (no lexical output): FAILS — the grammar must
    produce lexical content; a sequence consisting solely of terminal control
    is rejected.
6.  Final '<|im_end|>' is excluded from token_provenance_ratio denominator.
7.  Final '<|im_end|>' is excluded from surface-similarity decode.
8.  Generic angle-bracket strings (e.g. '<person>') are NOT classified as
    terminal control — unlicensed ones still fail normally.
9.  '<|endoftext|>' is NOT automatically a terminal control surface.
10. QWEN_TERMINAL_CONTROL_SURFACES is exported and contains exactly the
    expected set.
"""
from __future__ import annotations

import pytest

from constrained_translation.protocol import AlignedExample, GenerationResult
from constrained_translation.token_auditor import (
    QWEN_TERMINAL_CONTROL_SURFACES,
    TokenAuditor,
    TokenLicense,
)

# ---------------------------------------------------------------------------
# Shared mini-backend
# ---------------------------------------------------------------------------

IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


class _SimpleBackend:
    """Minimal backend: vocab list, IDs are indices."""

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = vocab

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            try:
                ids.append(self._vocab.index(word))
            except ValueError:
                pass
        return ids

    def decode_token(self, token_id: int) -> str:
        return self._vocab[token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Full-sequence detokenisation (space-join for this flat vocab)."""
        return " ".join(self._vocab[i] for i in token_ids)

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, grammar, max_tokens, temperature=0.0):  # pragma: no cover
        return GenerationResult(
            text="", token_ids=[], prompt_tokens=0, output_tokens=0, generation_ms=0.0
        )


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
# Shared fixture: vocab = ["hello", "world", "<|im_end|>"]
#   IDs: hello=0, world=1, <|im_end|>=2
#   License covers IDs 0 and 1 (from "hello world" example).
# ---------------------------------------------------------------------------

def _build_fixture():
    vocab = ["hello", "world", IM_END]
    backend = _SimpleBackend(vocab)
    auditor = TokenAuditor()
    ex = _make_example(verse_idx=0, target="hello world")
    licence: TokenLicense = auditor.build_license([ex], backend)
    attested_vocab = frozenset({"hello", "world"})
    return backend, auditor, licence, attested_vocab


# ════════════════════════════════════════════════════════════════════════════
# 1. QWEN_TERMINAL_CONTROL_SURFACES exported constant
# ════════════════════════════════════════════════════════════════════════════

class TestTerminalControlConstant:
    def test_constant_is_exported(self):
        """QWEN_TERMINAL_CONTROL_SURFACES must be importable from token_auditor."""
        # Import already happened at module level; assert it exists and is the right type.
        assert isinstance(QWEN_TERMINAL_CONTROL_SURFACES, frozenset), (
            "QWEN_TERMINAL_CONTROL_SURFACES must be a frozenset"
        )

    def test_constant_contains_im_end(self):
        """'<|im_end|>' must be in QWEN_TERMINAL_CONTROL_SURFACES."""
        assert IM_END in QWEN_TERMINAL_CONTROL_SURFACES, (
            f"Expected {IM_END!r} in QWEN_TERMINAL_CONTROL_SURFACES; "
            f"got: {QWEN_TERMINAL_CONTROL_SURFACES!r}"
        )

    def test_constant_does_not_contain_endoftext(self):
        """'<|endoftext|>' must NOT be in QWEN_TERMINAL_CONTROL_SURFACES by default.

        The spec says to add '<|endoftext|>' only if tests/config justify it.
        """
        assert ENDOFTEXT not in QWEN_TERMINAL_CONTROL_SURFACES, (
            f"<|endoftext|> must not be auto-added to QWEN_TERMINAL_CONTROL_SURFACES; "
            f"current set: {QWEN_TERMINAL_CONTROL_SURFACES!r}"
        )

    def test_generic_angle_brackets_not_in_constant(self):
        """Angle-bracket strings like '<person>' must NOT be in the set."""
        generic = "<person>"
        assert generic not in QWEN_TERMINAL_CONTROL_SURFACES


# ════════════════════════════════════════════════════════════════════════════
# 2. Final-position pass
# ════════════════════════════════════════════════════════════════════════════

class TestFinalTerminalControlPasses:

    def test_licensed_seq_plus_final_im_end_passes(self):
        """[hello, world, <|im_end|>] must pass: final control stripped cleanly."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        # IDs: hello=0, world=1, im_end=2
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, 1, ctrl_id]

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True, (
            f"Final <|im_end|> should be silently stripped; violations: {result.violations}"
        )
        assert result.violations == []

    def test_single_lexical_token_plus_final_im_end_passes(self):
        """[hello, <|im_end|>] — one licensed lexical token + final ctrl → pass."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, ctrl_id]  # hello + im_end

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True, (
            f"[hello, <|im_end|>] should pass; violations: {result.violations}"
        )

    def test_final_im_end_not_in_provenance_map(self):
        """The final terminal control must not appear in the returned provenance map."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, 1, ctrl_id]

        _, provenance = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert ctrl_id not in provenance, (
            f"Terminal control ID {ctrl_id} must not appear in provenance map; "
            f"got: {provenance}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. Non-final position — loud violation
# ════════════════════════════════════════════════════════════════════════════

class TestNonFinalTerminalControlFails:

    def test_im_end_at_start_fails(self):
        """[<|im_end|>, hello, world] — control at position 0 is a loud violation."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [ctrl_id, 0, 1]  # ctrl first → non-final → violation

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "<|im_end|> at non-final position must cause a failure"
        )
        # Violation message must name the surface and indicate position.
        assert any(IM_END in v for v in result.violations), (
            f"Expected violation mentioning {IM_END!r}; got: {result.violations}"
        )

    def test_im_end_in_middle_fails(self):
        """[hello, <|im_end|>, world] — control in the middle is a loud violation."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, ctrl_id, 1]  # ctrl in middle

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False
        assert any(IM_END in v for v in result.violations)

    def test_violation_message_mentions_position(self):
        """Violation string for non-final control should indicate it's at a non-final position."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [ctrl_id, 0]  # ctrl at index 0, non-final

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False
        # At least one violation should contain something indicating non-final usage.
        ctrl_violations = [v for v in result.violations if IM_END in v]
        assert ctrl_violations, f"Expected a violation for {IM_END!r}; got: {result.violations}"


# ════════════════════════════════════════════════════════════════════════════
# 4. Multiple terminal controls — only the very last can be final
# ════════════════════════════════════════════════════════════════════════════

class TestMultipleTerminalControls:

    def test_two_im_end_fails(self):
        """[hello, <|im_end|>, <|im_end|>] — first im_end is non-final → loud violation."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        # hello=0, im_end=2, im_end=2: second ctrl is final (ok), first is non-final (bad)
        token_ids = [0, ctrl_id, ctrl_id]

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "Two <|im_end|> tokens: earlier occurrence must fail loudly"
        )
        ctrl_violations = [v for v in result.violations if IM_END in v]
        assert ctrl_violations, (
            f"Expected a violation mentioning {IM_END!r}; got: {result.violations}"
        )

    def test_ctrl_sandwiched_between_lexical_fails(self):
        """[hello, <|im_end|>, world, <|im_end|>] — first ctrl is non-final → fails."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, ctrl_id, 1, ctrl_id]

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False
        assert any(IM_END in v for v in result.violations)


# ════════════════════════════════════════════════════════════════════════════
# 5. Control-only sequence fails (no lexical output)
# ════════════════════════════════════════════════════════════════════════════

class TestControlOnlySequenceFails:

    def test_im_end_only_sequence_fails(self):
        """[<|im_end|>] alone fails: no lexical output produced.

        The grammar requires lexical content; a sequence of solely the
        terminal control marker is a provenance failure (not a pass).
        """
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [ctrl_id]

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "A sequence consisting only of <|im_end|> must fail: no lexical output"
        )
        # There should be a violation explaining the lack of lexical output.
        assert result.violations, (
            "Expected at least one violation for control-only sequence"
        )

    def test_empty_sequence_still_passes(self):
        """Empty token_ids [] must still pass (unchanged existing behaviour)."""
        backend, auditor, licence, attested_vocab = _build_fixture()

        result, provenance = auditor.audit(
            token_ids=[],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is True
        assert result.violations == []
        assert provenance == {}


# ════════════════════════════════════════════════════════════════════════════
# 6. Scalar diagnostics — final control excluded from ratio/similarity
# ════════════════════════════════════════════════════════════════════════════

class TestScalarDiagnosticsExcludeTerminalControl:

    def test_final_im_end_does_not_lower_provenance_ratio(self):
        """token_provenance_ratio with [hello, world, im_end] == ratio with [hello, world].

        The terminal control must not count as an unlicensed token in the
        ratio denominator.
        """
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)

        result_without_ctrl, _ = auditor.audit(
            token_ids=[0, 1],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )
        result_with_ctrl, _ = auditor.audit(
            token_ids=[0, 1, ctrl_id],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result_with_ctrl.token_provenance_ratio == pytest.approx(
            result_without_ctrl.token_provenance_ratio
        ), (
            f"Final im_end must not lower token_provenance_ratio: "
            f"with_ctrl={result_with_ctrl.token_provenance_ratio}, "
            f"without_ctrl={result_without_ctrl.token_provenance_ratio}"
        )

    def test_final_im_end_does_not_lower_surface_similarity(self):
        """surface_attestation_similarity with [hello, world, im_end] == without ctrl."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)

        result_without_ctrl, _ = auditor.audit(
            token_ids=[0, 1],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )
        result_with_ctrl, _ = auditor.audit(
            token_ids=[0, 1, ctrl_id],
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result_with_ctrl.surface_attestation_similarity == pytest.approx(
            result_without_ctrl.surface_attestation_similarity
        ), (
            f"Final im_end must not lower surface_attestation_similarity: "
            f"with_ctrl={result_with_ctrl.surface_attestation_similarity}, "
            f"without_ctrl={result_without_ctrl.surface_attestation_similarity}"
        )

    def test_provenance_ratio_is_1_with_all_licensed_plus_final_ctrl(self):
        """All licensed tokens + final ctrl must yield token_provenance_ratio == 1.0."""
        backend, auditor, licence, attested_vocab = _build_fixture()
        ctrl_id = backend._vocab.index(IM_END)
        token_ids = [0, 1, ctrl_id]

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.token_provenance_ratio == pytest.approx(1.0), (
            f"Expected ratio 1.0; got {result.token_provenance_ratio}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. Generic angle-bracket strings are NOT terminal control
# ════════════════════════════════════════════════════════════════════════════

class TestGenericAngleBracketNotControl:

    def test_person_tag_not_classified_as_terminal_control(self):
        """'<person>' (unlicensed) must fail with an ID-level violation, not be silently passed.

        Generic angle-bracket strings are lexical text that happened to use angle
        brackets; they are NOT Qwen terminal control tokens.
        """
        vocab = ["hello", "<person>"]
        backend = _SimpleBackend(vocab)
        auditor = TokenAuditor()
        ex = _make_example(verse_idx=0, target="hello")
        licence: TokenLicense = auditor.build_license([ex], backend)
        attested_vocab = frozenset({"hello"})

        person_id = vocab.index("<person>")
        # "<person>" is not licensed AND not a terminal control
        token_ids = [0, person_id]  # hello + <person>

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "'<person>' must fail as an unlicensed token, not be silently accepted as control"
        )
        # Violation must mention the token_id (ID-level failure)
        assert any(str(person_id) in v for v in result.violations)

    def test_angle_bracket_token_at_end_not_silently_stripped(self):
        """An unlicensed angle-bracket string at the final position must still fail.

        Only '<|im_end|>' (and the named set in QWEN_TERMINAL_CONTROL_SURFACES)
        may be silently stripped; other angle-bracket tokens must not.
        """
        vocab = ["hello", "<end>"]
        backend = _SimpleBackend(vocab)
        auditor = TokenAuditor()
        ex = _make_example(verse_idx=0, target="hello")
        licence: TokenLicense = auditor.build_license([ex], backend)
        attested_vocab = frozenset({"hello"})

        end_id = vocab.index("<end>")
        token_ids = [0, end_id]  # hello + <end> — <end> is NOT in QWEN_TERMINAL_CONTROL_SURFACES

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "'<end>' must not be silently stripped — only the named terminal controls are"
        )


# ════════════════════════════════════════════════════════════════════════════
# 8. <|endoftext|> is NOT a terminal control surface
# ════════════════════════════════════════════════════════════════════════════

class TestEndoftextNotTerminalControl:

    def test_endoftext_unlicensed_fails(self):
        """'<|endoftext|>' is NOT in QWEN_TERMINAL_CONTROL_SURFACES by default.

        It must fail as an unlicensed ID, not be silently stripped.
        """
        vocab = ["hello", ENDOFTEXT]
        backend = _SimpleBackend(vocab)
        auditor = TokenAuditor()
        ex = _make_example(verse_idx=0, target="hello")
        licence: TokenLicense = auditor.build_license([ex], backend)
        attested_vocab = frozenset({"hello"})

        endoftext_id = vocab.index(ENDOFTEXT)
        token_ids = [0, endoftext_id]  # hello + <|endoftext|>

        result, _ = auditor.audit(
            token_ids=token_ids,
            attested_vocab=attested_vocab,
            backend=backend,
            license=licence,
        )

        assert result.passed is False, (
            "<|endoftext|> must not be silently stripped (not in QWEN_TERMINAL_CONTROL_SURFACES)"
        )
        assert any(str(endoftext_id) in v for v in result.violations)
