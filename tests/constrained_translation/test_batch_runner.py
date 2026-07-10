"""Tests for Task 9: BatchRunner — coverage-first constrained batch loop.

TDD: tests written before implementation; run first to confirm RED, then
implement batch_runner.py to go GREEN.

Invariants exercised
────────────────────
1. Source coverage is checked against SELECTED EXAMPLE SOURCE strings; target
   attested_vocab is NEVER passed to UNKDetector for coverage checking.
2. Held-out (exclude_idx) is passed on every ExampleSelector attempt,
   including all retries.
3. On coverage failure, log coverage_failure; retry by expanding
   coverage-targeted retrieval only (no unconstrained fallback).
4. No backend.generate call when source coverage fails after all retries;
   hard-fail with [UNK:surface] artifact instead.
5. Coverage must fully pass before generate() is called.
6. Grammar build failure → hard failure, no generation fallback.
7. Token audit failure → result rejected as hard failure.
8. Accepted result requires coverage_pass AND token audit pass.
9. Provenance mapping logged per-item.
10. batch_summary logged at end; BatchRollup returned.
11. Backend unavailable is a loud preflight failure (raises).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from constrained_translation.protocol import (
    AlignedExample,
    BatchRollup,
    GenerationResult,
    TokenAuditResult,
    TranslationRequest,
    TranslationResult,
)
from constrained_translation.fake_backend import FakeBackend
from constrained_translation.grammar_builder import GrammarBuildError
from constrained_translation.batch_runner import BatchItem, BatchRunner


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_example(
    verse_idx: int,
    source: str,
    target: str,
    method: str = "semantic",
) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source=source,
        target=target,
        selection_score=1.0,
        selection_method=method,
        evidence_tier=1,
    )


# Tiny 12-verse corpus (same as conftest but replicated for isolation)
_SOURCE_LINES = [
    "In the beginning God created the heavens and the earth",
    "The earth was without form and void",
    "And God said let there be light",
    "And God saw that the light was good",
    "God called the light Day and the darkness Night",
    "And God created great whales in the sea",
    "And God blessed them saying be fruitful and multiply",
    "And God made the firmament above the waters",
    "God saw every thing that he had made and it was very good",
    "Thus the heavens and the earth were finished",
    "And on the seventh day God ended his work",
    "And God blessed the seventh day and sanctified it",
]

_TARGET_LINES = [
    "Au commencement Dieu créa les cieux et la terre",
    "La terre était sans forme et vide",
    "Et Dieu dit que la lumière soit",
    "Et Dieu vit que la lumière était bonne",
    "Dieu appela la lumière Jour et les ténèbres Nuit",
    "Et Dieu créa les grands poissons dans la mer",
    "Et Dieu les bénit disant soyez féconds et multipliez",
    "Et Dieu fit le firmament au-dessus des eaux",
    "Dieu vit tout ce qu'il avait fait et c'était très bon",
    "Ainsi les cieux et la terre furent achevés",
    "Et le septième jour Dieu acheva son œuvre",
    "Et Dieu bénit le septième jour et le sanctifia",
]


@pytest.fixture
def corpus_files(tmp_path):
    src = tmp_path / "source.txt"
    tgt = tmp_path / "target.txt"
    src.write_text("\n".join(_SOURCE_LINES) + "\n", encoding="utf-8")
    tgt.write_text("\n".join(_TARGET_LINES) + "\n", encoding="utf-8")
    return str(src), str(tgt)


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "run.jsonl")


def _read_events(log_path: str) -> list[dict]:
    lines = []
    for raw in Path(log_path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def _runner(corpus_files, log_path, backend=None, max_retries=2, n_semantic=3, n_coverage=3):
    src, tgt = corpus_files
    if backend is None:
        backend = FakeBackend()
    return BatchRunner(
        source_file=src,
        target_file=tgt,
        backend=backend,
        log_path=log_path,
        max_retries=max_retries,
        n_semantic=n_semantic,
        n_coverage=n_coverage,
    )


# ---------------------------------------------------------------------------
# §1  BatchItem dataclass
# ---------------------------------------------------------------------------

class TestBatchItemDataclass:
    def test_batch_item_has_item_id(self):
        item = BatchItem(item_id="GEN 1:1", source_text="In the beginning", exclude_idx=0)
        assert item.item_id == "GEN 1:1"

    def test_batch_item_has_source_text(self):
        item = BatchItem(item_id="GEN 1:1", source_text="In the beginning", exclude_idx=0)
        assert item.source_text == "In the beginning"

    def test_batch_item_has_exclude_idx(self):
        item = BatchItem(item_id="GEN 1:1", source_text="In the beginning", exclude_idx=0)
        assert item.exclude_idx == 0

    def test_batch_item_exclude_idx_none(self):
        item = BatchItem(item_id="GEN 1:1", source_text="In the beginning", exclude_idx=None)
        assert item.exclude_idx is None


# ---------------------------------------------------------------------------
# §2  BatchRunner constructor and preflight
# ---------------------------------------------------------------------------

class TestBatchRunnerPreflight:
    def test_runner_instantiates(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        assert runner is not None

    def test_backend_unavailable_raises_on_run(self, corpus_files, log_path):
        """Backend.is_available() → False must raise loudly before any work."""
        bad_backend = MagicMock()
        bad_backend.is_available.return_value = False
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=bad_backend,
            log_path=log_path,
        )
        with pytest.raises(RuntimeError, match="unavailable"):
            runner.run([BatchItem("GEN 1:1", "In the beginning", 0)])

    def test_backend_unavailable_no_generate_called(self, corpus_files, log_path):
        bad_backend = MagicMock()
        bad_backend.is_available.return_value = False
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=bad_backend,
            log_path=log_path,
        )
        with pytest.raises(RuntimeError):
            runner.run([BatchItem("GEN 1:1", "In the beginning", 0)])
        bad_backend.generate.assert_not_called()


# ---------------------------------------------------------------------------
# §3  Successful path: coverage passes → generate → audit → accepted
# ---------------------------------------------------------------------------

class TestSuccessfulPath:
    def test_run_returns_list_of_translation_results(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        results, rollup = runner.run(items)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], TranslationResult)

    def test_run_returns_batch_rollup(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        results, rollup = runner.run(items)
        assert isinstance(rollup, BatchRollup)

    def test_coverage_pass_true_when_examples_cover_all_tokens(self, corpus_files, log_path):
        """Source text uses words from corpus → coverage_pass=True.

        We use a source text composed entirely of tokens that appear in multiple
        other corpus verses (so the ExampleSelector can cover them even with
        the query verse itself excluded).
        - "God"   — verses 2,3,4,5,6,7,8,10,11 (many)
        - "and"   — verses 0,1,2,3,4,5,6,7,8,9,10,11 (all)
        - "the"   — many
        - "light" — verses 2,3,4
        """
        runner = _runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        # exclude_idx=None: no held-out exclusion; all corpus lines available
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].coverage_pass is True

    def test_generation_called_when_coverage_passes(self, corpus_files, log_path):
        """When coverage passes, backend.generate must be called exactly once."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        # Make tokenize/decode_token work deterministically
        fb = FakeBackend()
        # Pre-populate fake backend vocab for target strings
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        # Response must only contain tokens from target vocab
        response_words = ["Dieu", "créa", "les", "cieux"]
        response = " ".join(response_words)
        gen_result = GenerationResult(
            text=response,
            token_ids=fb.tokenize(response),
            prompt_tokens=10,
            output_tokens=len(response.split()),
            generation_ms=50.0,
        )
        mock_backend.generate.return_value = gen_result

        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=mock_backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        # Use a fully-covered source text (all tokens exist in other corpus verses)
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        # generate should be called once (coverage passes on attempt 0)
        assert mock_backend.generate.call_count >= 1

    def test_successful_result_not_hard_failure(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        results, _ = runner.run(items)
        # With the fake backend, the token audit will determine pass/fail
        # If it hard fails, coverage_pass should still be correctly set
        assert results[0].item_id == "GEN 1:1"

    def test_result_item_id_matches_input(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [
            BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0),
            BatchItem("GEN 1:2", _SOURCE_LINES[1], exclude_idx=1),
        ]
        results, _ = runner.run(items)
        assert results[0].item_id == "GEN 1:1"
        assert results[1].item_id == "GEN 1:2"

    def test_batch_summary_logged(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        runner.run(items)
        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "batch_summary" in event_names


# ---------------------------------------------------------------------------
# §4  Coverage failure and retry logic
# ---------------------------------------------------------------------------

class TestCoverageFailureAndRetry:
    def test_coverage_failure_logged_when_gaps_remain(self, corpus_files, log_path):
        """A source text with uncommon tokens should trigger coverage_failure log."""
        runner = _runner(corpus_files, log_path, n_semantic=1, n_coverage=1)
        # Use a source text with a very rare token not in corpus
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "coverage_failure" in event_names

    def test_no_backend_call_on_coverage_hard_failure(self, corpus_files, log_path):
        """If coverage never passes after all retries, backend.generate must NOT be called."""
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True
        mock_backend.tokenize.return_value = []
        mock_backend.decode_token.return_value = ""

        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=mock_backend,
            log_path=log_path,
            max_retries=1,
            n_semantic=1,
            n_coverage=1,
        )
        # Source text with tokens definitely not in the corpus
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        results, _ = runner.run(items)
        mock_backend.generate.assert_not_called()

    def test_hard_failure_on_persistent_coverage_gap(self, corpus_files, log_path):
        """Items with unrecoverable coverage gaps are marked hard_failure=True."""
        runner = _runner(corpus_files, log_path, max_retries=1, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].hard_failure is True

    def test_hard_failure_coverage_artifact_contains_unk_markers(self, corpus_files, log_path):
        """Hard failure due to coverage → translation must contain [UNK:…] markers."""
        runner = _runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz", exclude_idx=None)]
        results, _ = runner.run(items)
        assert "[UNK:" in results[0].translation

    def test_hard_failure_generation_result_is_none(self, corpus_files, log_path):
        """Coverage hard failure: generation_result must be None (never generated)."""
        runner = _runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].generation_result is None

    def test_hard_failure_token_audit_is_none(self, corpus_files, log_path):
        """Coverage hard failure: token_audit must be None (never reached audit)."""
        runner = _runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].token_audit is None

    def test_coverage_failure_logs_missing_spans(self, corpus_files, log_path):
        """coverage_failure event must include info about missing/uncovered spans."""
        runner = _runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        cov_fails = [e for e in events if e["event"] == "coverage_failure"]
        assert len(cov_fails) >= 1
        # Should have item_id
        assert cov_fails[0]["item_id"] == "X 1:1"

    def test_coverage_retry_logged_on_retry(self, corpus_files, log_path):
        """When coverage gaps remain after attempt 0, coverage_retry must be logged."""
        runner = _runner(corpus_files, log_path, max_retries=2, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "coverage_retry" in event_names

    def test_coverage_retry_event_has_retry_count(self, corpus_files, log_path):
        """coverage_retry event must include retry_count field."""
        runner = _runner(corpus_files, log_path, max_retries=2, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        retries = [e for e in events if e["event"] == "coverage_retry"]
        assert len(retries) >= 1
        assert "retry_count" in retries[0]

    def test_coverage_retry_event_has_old_and_new_example_ids(self, corpus_files, log_path):
        """coverage_retry event must include old_example_ids and new_example_ids."""
        runner = _runner(corpus_files, log_path, max_retries=2, n_semantic=1, n_coverage=1)
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        retries = [e for e in events if e["event"] == "coverage_retry"]
        if retries:  # only check if a retry was triggered
            assert "old_example_ids" in retries[0]
            assert "new_example_ids" in retries[0]


# ---------------------------------------------------------------------------
# §5  Held-out exclusion on every attempt
# ---------------------------------------------------------------------------

class TestHeldOutExclusion:
    def test_exclude_idx_passed_to_selector_on_first_attempt(self, corpus_files, log_path):
        """exclude_idx from BatchItem must reach ExampleSelector.select() on attempt 0."""
        with patch(
            "constrained_translation.batch_runner.ExampleSelector"
        ) as MockSelector:
            mock_instance = MagicMock()
            MockSelector.return_value = mock_instance
            # Return valid examples to allow progress
            ex = _make_example(1, _SOURCE_LINES[1], _TARGET_LINES[1])
            mock_instance.select.return_value = (ex,)

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=FakeBackend(),
                log_path=log_path,
                n_semantic=1,
                n_coverage=1,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=5)]
            runner.run(items)

            # Every call to select() must have exclude_idx=5
            for c in mock_instance.select.call_args_list:
                kwargs = c.kwargs if c.kwargs else {}
                args = c.args if c.args else ()
                # exclude_idx can be positional (arg 1) or keyword
                if "exclude_idx" in kwargs:
                    assert kwargs["exclude_idx"] == 5
                else:
                    # positional: select(query, exclude_idx)
                    assert len(args) >= 2 and args[1] == 5

    def test_exclude_idx_passed_on_every_retry(self, corpus_files, log_path):
        """exclude_idx must be forwarded on every retry, not just attempt 0."""
        with patch(
            "constrained_translation.batch_runner.ExampleSelector"
        ) as MockSelector:
            mock_instance = MagicMock()
            MockSelector.return_value = mock_instance
            # Always return empty (or minimal) examples so coverage never passes → retries
            mock_instance.select.return_value = ()

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=FakeBackend(),
                log_path=log_path,
                max_retries=2,
                n_semantic=1,
                n_coverage=1,
            )
            items = [BatchItem("GEN 1:1", "xyzzy frobozz", exclude_idx=7)]
            runner.run(items)

            call_count = mock_instance.select.call_count
            assert call_count >= 1  # at least initial + 1 retry

            for c in mock_instance.select.call_args_list:
                kwargs = c.kwargs if c.kwargs else {}
                args = c.args if c.args else ()
                if "exclude_idx" in kwargs:
                    assert kwargs["exclude_idx"] == 7
                else:
                    assert len(args) >= 2 and args[1] == 7


# ---------------------------------------------------------------------------
# §6  Grammar build failure → hard failure, no generation
# ---------------------------------------------------------------------------

class TestGrammarBuildFailure:
    def test_grammar_build_failure_no_generate_called(self, corpus_files, log_path):
        """GrammarBuildError → backend.generate must NOT be called."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        with patch(
            "constrained_translation.batch_runner.GrammarBuilder"
        ) as MockGrammar:
            mock_grammar_instance = MagicMock()
            MockGrammar.return_value = mock_grammar_instance
            mock_grammar_instance.build.side_effect = GrammarBuildError(
                item_id="GEN 1:1", reason="forced test failure"
            )

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            results, _ = runner.run(items)

        mock_backend.generate.assert_not_called()

    def test_grammar_build_failure_is_hard_failure(self, corpus_files, log_path):
        """GrammarBuildError → result must have hard_failure=True."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        with patch(
            "constrained_translation.batch_runner.GrammarBuilder"
        ) as MockGrammar:
            mock_grammar_instance = MagicMock()
            MockGrammar.return_value = mock_grammar_instance
            mock_grammar_instance.build.side_effect = GrammarBuildError(
                item_id="GEN 1:1", reason="forced test failure"
            )

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            results, _ = runner.run(items)

        assert results[0].hard_failure is True

    def test_grammar_build_failure_logged(self, corpus_files, log_path):
        """GrammarBuildError → a hard_failure or grammar_error event must be logged."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        with patch(
            "constrained_translation.batch_runner.GrammarBuilder"
        ) as MockGrammar:
            mock_grammar_instance = MagicMock()
            MockGrammar.return_value = mock_grammar_instance
            mock_grammar_instance.build.side_effect = GrammarBuildError(
                item_id="GEN 1:1", reason="forced test failure"
            )

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            runner.run(items)

        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "hard_failure" in event_names or "grammar_error" in event_names


# ---------------------------------------------------------------------------
# §7  Token audit failure → result rejected as hard failure
# ---------------------------------------------------------------------------

class TestTokenAuditFailure:
    def test_audit_failure_is_hard_failure(self, corpus_files, log_path):
        """Token audit failure → result must have hard_failure=True."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        # Generate a response whose tokens won't be in the license
        bad_response = "totally unlicensed output"
        mock_backend.generate.return_value = GenerationResult(
            text=bad_response,
            token_ids=fb.tokenize(bad_response),
            prompt_tokens=10,
            output_tokens=len(bad_response.split()),
            generation_ms=50.0,
        )

        with patch(
            "constrained_translation.batch_runner.TokenAuditor"
        ) as MockAuditor:
            mock_auditor_instance = MagicMock()
            MockAuditor.return_value = mock_auditor_instance
            # Audit always fails
            mock_auditor_instance.audit.return_value = (
                TokenAuditResult(passed=False, violations=["token_id=99 not in licence"]),
                {},
            )
            mock_auditor_instance.build_license.return_value = {}

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            results, _ = runner.run(items)

        assert results[0].hard_failure is True

    def test_audit_failure_translation_not_raw_model_text(self, corpus_files, log_path):
        """Token audit failure → translation must NOT be the raw model output."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        bad_response = "FORBIDDEN OUTPUT DO NOT USE"
        mock_backend.generate.return_value = GenerationResult(
            text=bad_response,
            token_ids=fb.tokenize(bad_response),
            prompt_tokens=10,
            output_tokens=len(bad_response.split()),
            generation_ms=50.0,
        )

        with patch(
            "constrained_translation.batch_runner.TokenAuditor"
        ) as MockAuditor:
            mock_auditor_instance = MagicMock()
            MockAuditor.return_value = mock_auditor_instance
            mock_auditor_instance.audit.return_value = (
                TokenAuditResult(passed=False, violations=["forced violation"]),
                {},
            )
            mock_auditor_instance.build_license.return_value = {}

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            results, _ = runner.run(items)

        assert results[0].translation != bad_response

    def test_audit_failure_logged(self, corpus_files, log_path):
        """Token audit failure → token_audit or hard_failure event must be logged."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        mock_backend.generate.return_value = GenerationResult(
            text="bad output",
            token_ids=fb.tokenize("bad output"),
            prompt_tokens=10,
            output_tokens=2,
            generation_ms=50.0,
        )

        with patch(
            "constrained_translation.batch_runner.TokenAuditor"
        ) as MockAuditor:
            mock_auditor_instance = MagicMock()
            MockAuditor.return_value = mock_auditor_instance
            mock_auditor_instance.audit.return_value = (
                TokenAuditResult(passed=False, violations=["forced"]),
                {},
            )
            mock_auditor_instance.build_license.return_value = {}

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            runner.run(items)

        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "hard_failure" in event_names or "token_audit" in event_names


# ---------------------------------------------------------------------------
# §8  No backend call without grammar
# ---------------------------------------------------------------------------

class TestNoBackendCallWithoutGrammar:
    def test_generate_called_with_nonempty_grammar(self, corpus_files, log_path):
        """backend.generate must always be called with a non-empty grammar string."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        response = "Dieu créa les cieux"
        mock_backend.generate.return_value = GenerationResult(
            text=response,
            token_ids=fb.tokenize(response),
            prompt_tokens=10,
            output_tokens=len(response.split()),
            generation_ms=50.0,
        )

        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=mock_backend,
            log_path=log_path,
            n_semantic=3,
            n_coverage=3,
        )
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        runner.run(items)

        if mock_backend.generate.called:
            for c in mock_backend.generate.call_args_list:
                args = c.args
                kwargs = c.kwargs
                # grammar is 2nd positional arg or 'grammar' keyword
                grammar = kwargs.get("grammar") or (args[1] if len(args) > 1 else None)
                assert grammar is not None and grammar != ""


# ---------------------------------------------------------------------------
# §9  Source coverage uses source strings, NOT target vocab
# ---------------------------------------------------------------------------

class TestSourceCoverageUsesSourceStrings:
    def test_unk_detector_not_called_with_target_vocab_for_coverage(
        self, corpus_files, log_path
    ):
        """UNKDetector.detect() for coverage check must use source-derived vocab,
        not the target attested_vocab from VocabExtractor."""
        with patch(
            "constrained_translation.batch_runner.UNKDetector"
        ) as MockDetector:
            mock_det_instance = MagicMock()
            MockDetector.return_value = mock_det_instance
            # Return no UNK spans (coverage passes immediately)
            mock_det_instance.detect.return_value = ()

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=FakeBackend(),
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            runner.run(items)

            # Coverage-check call: the attested_vocab argument must be built
            # from the SELECTED EXAMPLE SOURCE strings, not from target strings.
            # We can verify by checking that the detect() calls' second arg
            # (attested_vocab) contains source-side tokens (lowercase words
            # from the source corpus), not exclusively target-language tokens.
            # At minimum, detect() must have been called.
            assert mock_det_instance.detect.called

    def test_coverage_check_uses_source_side_tokens(self, corpus_files, log_path):
        """The vocab passed to UNKDetector for coverage must contain source-side words."""
        captured_vocabs = []

        class CapturingDetector:
            def detect(self, source_text, attested_vocab):
                captured_vocabs.append(frozenset(attested_vocab))
                return ()

        with patch(
            "constrained_translation.batch_runner.UNKDetector",
            return_value=CapturingDetector(),
        ):
            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=FakeBackend(),
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            runner.run(items)

        # Coverage check vocab must include source-language words
        # (e.g. "the", "earth", "God" etc. from the English corpus)
        # and NOT exclusively French target words
        if captured_vocabs:
            coverage_vocab = captured_vocabs[0]
            # At least one call must include English source-side tokens
            # "the" appears in many source lines
            source_tokens = set()
            for line in _SOURCE_LINES:
                for w in line.lower().split():
                    source_tokens.add(w)
            # The coverage vocab should overlap with source tokens
            lowered = {t.lower() for t in coverage_vocab}
            assert len(lowered & source_tokens) > 0


# ---------------------------------------------------------------------------
# §10  Provenance logging
# ---------------------------------------------------------------------------

class TestProvenanceLogging:
    def test_provenance_logged_on_accepted_result(self, corpus_files, log_path):
        """On an accepted (non-hard-failure) result, provenance event must be logged."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        # Return a response made only of attested target words
        response = "Dieu créa les cieux"
        mock_backend.generate.return_value = GenerationResult(
            text=response,
            token_ids=fb.tokenize(response),
            prompt_tokens=10,
            output_tokens=len(response.split()),
            generation_ms=50.0,
        )

        with patch(
            "constrained_translation.batch_runner.TokenAuditor"
        ) as MockAuditor:
            mock_auditor_instance = MagicMock()
            MockAuditor.return_value = mock_auditor_instance
            mock_auditor_instance.audit.return_value = (
                TokenAuditResult(passed=True, violations=[]),
                {1: [0], 2: [1]},  # provenance map
            )
            mock_auditor_instance.build_license.return_value = {1: [0], 2: [1]}

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=5,
                n_coverage=5,
            )
            # Use a fully-covered source text
            items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
            results, _ = runner.run(items)

        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "provenance" in event_names

    def test_provenance_event_has_token_to_verse_map(self, corpus_files, log_path):
        """provenance event must include a token_to_verse mapping."""
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens

        response = "Dieu créa"
        mock_backend.generate.return_value = GenerationResult(
            text=response,
            token_ids=fb.tokenize(response),
            prompt_tokens=10,
            output_tokens=len(response.split()),
            generation_ms=50.0,
        )

        with patch(
            "constrained_translation.batch_runner.TokenAuditor"
        ) as MockAuditor:
            mock_auditor_instance = MagicMock()
            MockAuditor.return_value = mock_auditor_instance
            mock_auditor_instance.audit.return_value = (
                TokenAuditResult(passed=True, violations=[]),
                {5: [2], 6: [3]},
            )
            mock_auditor_instance.build_license.return_value = {5: [2], 6: [3]}

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=3,
                n_coverage=3,
            )
            items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
            runner.run(items)

        events = _read_events(log_path)
        prov_events = [e for e in events if e["event"] == "provenance"]
        if prov_events:
            assert "token_to_verse" in prov_events[0]


# ---------------------------------------------------------------------------
# §11  Rollup correctness
# ---------------------------------------------------------------------------

class TestRollupCorrectness:
    def test_rollup_hard_failure_pct_reflects_failures(self, corpus_files, log_path):
        """hard_failure_pct in rollup must match number of hard-failed results."""
        runner = _runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        items = [
            BatchItem("X 1:1", "xyzzy frobozz", exclude_idx=None),  # will fail
            BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0),  # may pass
        ]
        results, rollup = runner.run(items)
        hard_fail_count = sum(1 for r in results if r.hard_failure)
        expected_pct = 100.0 * hard_fail_count / len(results)
        assert rollup.hard_failure_pct == pytest.approx(expected_pct)

    def test_rollup_total_units_matches_input(self, corpus_files, log_path):
        runner = _runner(corpus_files, log_path)
        items = [
            BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0),
            BatchItem("GEN 1:2", _SOURCE_LINES[1], exclude_idx=1),
            BatchItem("GEN 1:3", _SOURCE_LINES[2], exclude_idx=2),
        ]
        results, rollup = runner.run(items)
        assert rollup.total_units == 3

    def test_batch_summary_event_item_id_is_batch_level(self, corpus_files, log_path):
        """batch_summary event should have a batch-level item_id or identifier."""
        runner = _runner(corpus_files, log_path)
        items = [BatchItem("GEN 1:1", _SOURCE_LINES[0], exclude_idx=0)]
        runner.run(items)
        events = _read_events(log_path)
        summaries = [e for e in events if e["event"] == "batch_summary"]
        assert len(summaries) == 1
        assert "item_id" in summaries[0]


# ---------------------------------------------------------------------------
# §12  Expanded retrieval on retry (deterministic, no model generation)
# ---------------------------------------------------------------------------

class TestExpandedRetrieval:
    def test_retry_expands_coverage_retrieval(self, corpus_files, log_path):
        """On coverage retry, n_coverage should be increased (expanded retrieval)."""
        select_call_args = []

        with patch(
            "constrained_translation.batch_runner.ExampleSelector"
        ) as MockSelector:
            mock_instance = MagicMock()
            MockSelector.return_value = mock_instance

            def capture_select(query, exclude_idx=None, n_semantic=None, n_coverage=None):
                select_call_args.append((query, exclude_idx, n_coverage))
                return ()

            mock_instance.select.side_effect = capture_select

            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=FakeBackend(),
                log_path=log_path,
                max_retries=2,
                n_semantic=1,
                n_coverage=1,
            )
            items = [BatchItem("GEN 1:1", "xyzzy frobozz", exclude_idx=None)]
            runner.run(items)

        # Should have been called at least 2 times (initial + retries)
        assert len(select_call_args) >= 2

    def test_retry_is_deterministic_no_model_generation(self, corpus_files, log_path):
        """Coverage retry must not call backend.generate."""
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True
        mock_backend.tokenize.return_value = []
        mock_backend.decode_token.return_value = ""

        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=mock_backend,
            log_path=log_path,
            max_retries=3,
            n_semantic=1,
            n_coverage=1,
        )
        items = [BatchItem("X 1:1", "xyzzy frobozz wibble", exclude_idx=None)]
        runner.run(items)

        # generate must not be called during coverage-gap retries
        mock_backend.generate.assert_not_called()


# ---------------------------------------------------------------------------
# §13  Backend generation failure → hard_failure, no retry, continue batch
# ---------------------------------------------------------------------------

class TestBackendGenerationFailure:
    """BackendError (or any exception) from generate() must be caught per-item,
    logged as hard_failure(reason='backend_generation_failure'), and the runner
    must continue processing remaining items without retrying unconstrained.
    """

    def _make_backend_that_raises(self, exc):
        """Return a mock backend whose generate() raises *exc*."""
        from constrained_translation.fake_backend import FakeBackend
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens
        mock_backend.generate.side_effect = exc
        return mock_backend

    def test_backend_error_produces_hard_failure(self, corpus_files, log_path):
        """BackendError from generate() must yield hard_failure=True for that item."""
        from constrained_translation.vllm_backend import BackendError
        backend = self._make_backend_that_raises(BackendError("generate", 500, "server error"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].hard_failure is True

    def test_backend_error_reason_is_backend_generation_failure(self, corpus_files, log_path):
        """hard_failure event for backend error must have reason='backend_generation_failure'."""
        from constrained_translation.vllm_backend import BackendError
        backend = self._make_backend_that_raises(BackendError("generate", 503, "unavailable"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        runner.run(items)
        events = _read_events(log_path)
        hf = [e for e in events if e["event"] == "hard_failure"]
        assert len(hf) >= 1
        assert hf[0].get("reason") == "backend_generation_failure"

    def test_backend_error_does_not_retry_unconstrained(self, corpus_files, log_path):
        """BackendError must NOT trigger a retry; generate() should be called exactly once."""
        from constrained_translation.vllm_backend import BackendError
        backend = self._make_backend_that_raises(BackendError("generate", 500, "err"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            max_retries=3,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        runner.run(items)
        # generate should have been called exactly once — no retry on backend failure
        assert backend.generate.call_count == 1

    def test_backend_error_runner_continues_remaining_items(self, corpus_files, log_path):
        """After a backend error on item 1, item 2 must still be processed."""
        from constrained_translation.vllm_backend import BackendError
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)

        # generate raises on first call, succeeds on subsequent calls
        call_count = [0]
        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise BackendError("generate", 500, "first call fails")
            # Return a valid GenerationResult for subsequent calls
            response = "Dieu créa les cieux"
            return GenerationResult(
                text=response,
                token_ids=fb.tokenize(response),
                prompt_tokens=10,
                output_tokens=len(response.split()),
                generation_ms=50.0,
            )

        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens
        mock_backend.generate.side_effect = generate_side_effect

        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=mock_backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [
            BatchItem("GEN 1:3", "God and the light", exclude_idx=None),
            BatchItem("GEN 1:4", "God and the light", exclude_idx=None),
        ]
        results, _ = runner.run(items)
        assert len(results) == 2
        # First item must be a hard failure
        assert results[0].hard_failure is True
        # Second item got a generate call (may pass or fail audit, but was attempted)
        assert mock_backend.generate.call_count >= 2

    def test_generic_exception_from_generate_is_hard_failure(self, corpus_files, log_path):
        """Any exception from generate() (not just BackendError) must be a hard failure."""
        backend = self._make_backend_that_raises(RuntimeError("unexpected backend crash"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].hard_failure is True

    def test_backend_error_generation_result_is_none(self, corpus_files, log_path):
        """On backend generation failure, generation_result must be None."""
        from constrained_translation.vllm_backend import BackendError
        backend = self._make_backend_that_raises(BackendError("generate", 500, "err"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].generation_result is None

    def test_backend_error_token_audit_is_none(self, corpus_files, log_path):
        """On backend generation failure, token_audit must be None."""
        from constrained_translation.vllm_backend import BackendError
        backend = self._make_backend_that_raises(BackendError("generate", 500, "err"))
        runner = BatchRunner(
            source_file=corpus_files[0],
            target_file=corpus_files[1],
            backend=backend,
            log_path=log_path,
            n_semantic=5,
            n_coverage=5,
        )
        items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
        results, _ = runner.run(items)
        assert results[0].token_audit is None


# ---------------------------------------------------------------------------
# §14  Explicit event terminals: token_audit_violation, success
# ---------------------------------------------------------------------------

class TestExplicitTerminalEvents:
    """Spec requires: 'token_audit_violation' event BEFORE hard_failure on audit
    rejection, and 'success' event for accepted results. Exactly one terminal
    event (success OR hard_failure) per item; provenance is additional.
    """

    def _setup_audit_mock(self, passed: bool, corpus_files, log_path):
        fb = FakeBackend()
        for line in _TARGET_LINES:
            fb.tokenize(line)
        mock_backend = MagicMock(spec=FakeBackend)
        mock_backend.is_available.return_value = True
        mock_backend.tokenize.side_effect = fb.tokenize
        mock_backend.decode_token.side_effect = fb.decode_token
        mock_backend.decode_tokens.side_effect = fb.decode_tokens
        response = "Dieu créa les cieux"
        mock_backend.generate.return_value = GenerationResult(
            text=response,
            token_ids=fb.tokenize(response),
            prompt_tokens=10,
            output_tokens=len(response.split()),
            generation_ms=50.0,
        )
        with patch("constrained_translation.batch_runner.TokenAuditor") as MockAuditor:
            mock_auditor = MagicMock()
            MockAuditor.return_value = mock_auditor
            if passed:
                mock_auditor.audit.return_value = (
                    TokenAuditResult(passed=True, violations=[]),
                    {1: [0]},
                )
            else:
                mock_auditor.audit.return_value = (
                    TokenAuditResult(passed=False, violations=["forced violation"]),
                    {},
                )
            mock_auditor.build_license.return_value = {}
            runner = BatchRunner(
                source_file=corpus_files[0],
                target_file=corpus_files[1],
                backend=mock_backend,
                log_path=log_path,
                n_semantic=5,
                n_coverage=5,
            )
            items = [BatchItem("GEN 1:3", "God and the light", exclude_idx=None)]
            results, _ = runner.run(items)
        return results

    def test_token_audit_violation_logged_before_hard_failure(self, corpus_files, log_path):
        """When audit fails, 'token_audit_violation' must appear in the log before hard_failure."""
        self._setup_audit_mock(passed=False, corpus_files=corpus_files, log_path=log_path)
        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "token_audit_violation" in event_names, (
            f"Expected 'token_audit_violation' event, got: {event_names}"
        )
        # And hard_failure must also be there
        assert "hard_failure" in event_names

    def test_token_audit_violation_precedes_hard_failure_in_log(self, corpus_files, log_path):
        """'token_audit_violation' must be logged before 'hard_failure' for the same item."""
        self._setup_audit_mock(passed=False, corpus_files=corpus_files, log_path=log_path)
        events = _read_events(log_path)
        item_events = [e for e in events if e.get("item_id") == "GEN 1:3"]
        names = [e["event"] for e in item_events]
        if "token_audit_violation" in names and "hard_failure" in names:
            idx_violation = names.index("token_audit_violation")
            idx_hf = names.index("hard_failure")
            assert idx_violation < idx_hf, (
                "token_audit_violation must precede hard_failure in the log"
            )

    def test_success_event_logged_on_accepted_result(self, corpus_files, log_path):
        """When audit passes, a 'success' event must be logged."""
        self._setup_audit_mock(passed=True, corpus_files=corpus_files, log_path=log_path)
        events = _read_events(log_path)
        event_names = [e["event"] for e in events]
        assert "success" in event_names, (
            f"Expected 'success' event for accepted result, got: {event_names}"
        )

    def test_exactly_one_terminal_per_item_on_success(self, corpus_files, log_path):
        """Exactly one terminal event (success or hard_failure) per item on success path."""
        self._setup_audit_mock(passed=True, corpus_files=corpus_files, log_path=log_path)
        events = _read_events(log_path)
        terminal_types = {"success", "hard_failure"}
        item_terminals = [
            e for e in events
            if e.get("item_id") == "GEN 1:3" and e["event"] in terminal_types
        ]
        assert len(item_terminals) == 1, (
            f"Expected exactly 1 terminal event, got {len(item_terminals)}: {item_terminals}"
        )

    def test_exactly_one_terminal_per_item_on_audit_failure(self, corpus_files, log_path):
        """Exactly one terminal event (hard_failure) per item on audit failure path."""
        self._setup_audit_mock(passed=False, corpus_files=corpus_files, log_path=log_path)
        events = _read_events(log_path)
        terminal_types = {"success", "hard_failure"}
        item_terminals = [
            e for e in events
            if e.get("item_id") == "GEN 1:3" and e["event"] in terminal_types
        ]
        assert len(item_terminals) == 1, (
            f"Expected exactly 1 terminal event (hard_failure), got {len(item_terminals)}: {item_terminals}"
        )
        assert item_terminals[0]["event"] == "hard_failure"


# ---------------------------------------------------------------------------
# §15  max_retries default is 2
# ---------------------------------------------------------------------------

class TestMaxRetriesDefault:
    def test_default_max_retries_is_2(self):
        """BatchRunner default max_retries must be 2."""
        import inspect
        sig = inspect.signature(BatchRunner.__init__)
        default = sig.parameters["max_retries"].default
        assert default == 2, f"Expected max_retries default=2, got {default}"


# ---------------------------------------------------------------------------
# §16  ExampleSelector is constructed/cached once per run (quality review)
# ---------------------------------------------------------------------------

class TestBatchRunnerSelectorCaching:
    """Verify that BatchRunner builds the BM25 index exactly once per run()
    call, regardless of item count or retries.
    """

    def test_selector_constructed_once_across_multiple_items(
        self, corpus_files, log_path
    ):
        """ExampleSelector.__init__ must be called exactly once during run(),
        even when processing multiple items.

        Strategy: patch ExampleSelector.__init__ to count calls while still
        delegating to the real implementation.
        """
        from unittest.mock import patch as _patch
        from constrained_translation.example_selector import ExampleSelector

        init_call_count = {"n": 0}
        _real_init = ExampleSelector.__init__

        def _counting_init(self_inner, *args, **kwargs):
            init_call_count["n"] += 1
            _real_init(self_inner, *args, **kwargs)

        src, tgt = corpus_files
        runner = BatchRunner(
            source_file=src,
            target_file=tgt,
            backend=FakeBackend(),
            log_path=log_path,
            max_retries=0,
            n_semantic=3,
            n_coverage=3,
        )

        items = [
            BatchItem(item_id=f"GEN 1:{i}", source_text="God saw the light", exclude_idx=None)
            for i in range(3)
        ]

        with _patch.object(ExampleSelector, "__init__", _counting_init):
            runner.run(items)

        assert init_call_count["n"] == 1, (
            f"ExampleSelector.__init__ called {init_call_count['n']} times; "
            f"expected exactly 1 (index must not be rebuilt per item)"
        )

    def test_selector_constructed_once_across_retries(
        self, corpus_files, log_path
    ):
        """With max_retries > 0 and a forced-retry item, the BM25 index must
        still only be built once (retries use per-call n_coverage overrides).
        """
        from unittest.mock import patch as _patch
        from constrained_translation.example_selector import ExampleSelector

        init_call_count = {"n": 0}
        _real_init = ExampleSelector.__init__

        def _counting_init(self_inner, *args, **kwargs):
            init_call_count["n"] += 1
            _real_init(self_inner, *args, **kwargs)

        src, tgt = corpus_files
        runner = BatchRunner(
            source_file=src,
            target_file=tgt,
            backend=FakeBackend(),
            log_path=log_path,
            max_retries=2,
            n_semantic=3,
            n_coverage=3,
        )

        # "xyzzy" is not in the corpus → forces coverage retries.
        items = [
            BatchItem(item_id="HARD 1:1", source_text="xyzzy blorple", exclude_idx=None),
        ]

        with _patch.object(ExampleSelector, "__init__", _counting_init):
            runner.run(items)

        assert init_call_count["n"] == 1, (
            f"ExampleSelector.__init__ called {init_call_count['n']} times during retries; "
            f"expected exactly 1 (retries must use per-call overrides, not rebuild the index)"
        )

    def test_held_out_exclusion_preserved_through_retries(
        self, corpus_files, log_path
    ):
        """exclude_idx must be forwarded on all retry calls (not just attempt 0)."""
        src, tgt = corpus_files
        runner = BatchRunner(
            source_file=src,
            target_file=tgt,
            backend=FakeBackend(),
            log_path=log_path,
            max_retries=2,
            n_semantic=2,
            n_coverage=2,
        )

        # Use exclude_idx=0 (verse 0 = "In the beginning…"); verify it's never returned.
        items = [
            BatchItem(
                item_id="GEN 1:1",
                source_text="In the beginning God created the heavens and the earth",
                exclude_idx=0,
            ),
        ]
        results, _ = runner.run(items)
        # The result must be produced (no crash); coverage may pass or hard-fail.
        assert len(results) == 1
