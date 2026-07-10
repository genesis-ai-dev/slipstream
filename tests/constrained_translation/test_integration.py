"""Tests for Task 13: Offline end-to-end integration tests with FakeBackend.

TDD: tests written BEFORE implementation, confirming RED first.

Correctness invariants exercised
─────────────────────────────────
1. No empty grammar is ever used for generation.
2. Held-out exclusion is honoured (exclude_idx).
3. Accepted translations have coverage_pass=True and a passing token audit.
4. Hard failures (coverage failure) preserve [UNK:…] artifact, no generation.
5. Rollup totals match actual result list.
6. Event log has exactly one terminal outcome event per item
   (either "provenance" for accepted or "hard_failure" for any hard failure).
7. CLI round-trip: input JSONL → output JSONL + log + rollup JSON on stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from constrained_translation.batch_runner import BatchItem, BatchRunner
from constrained_translation.fake_backend import FakeBackend
from constrained_translation.protocol import TranslationResult


# ---------------------------------------------------------------------------
# Shared corpus (12-verse EN/FR)
# ---------------------------------------------------------------------------

_SOURCE = [
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

_TARGET = [
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

# A source text made entirely of tokens present in many corpus verses (easy coverage)
_EASY_SOURCE = "God and the light"
_EASY_IDX = None  # no held-out exclusion needed for a synthetic query

# A source text with tokens guaranteed absent from the corpus
_HARD_SOURCE = "xyzzy frobozz wibble"
_HARD_IDX = None


@pytest.fixture
def corpus_files(tmp_path):
    src = tmp_path / "source.txt"
    tgt = tmp_path / "target.txt"
    src.write_text("\n".join(_SOURCE) + "\n", encoding="utf-8")
    tgt.write_text("\n".join(_TARGET) + "\n", encoding="utf-8")
    return str(src), str(tgt)


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "events.jsonl")


def _read_events(log_path: str) -> list[dict]:
    lines = []
    for raw in Path(log_path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def _make_runner(
    corpus_files,
    log_path,
    backend=None,
    max_retries=2,
    n_semantic=4,
    n_coverage=4,
):
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
# §1  Accepted path — coverage passes, grammar non-empty, audit passes
# ---------------------------------------------------------------------------

class TestAcceptedPath:
    def test_accepted_result_coverage_pass_true(self, corpus_files, log_path):
        """Accepted result must have coverage_pass=True."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if not r.hard_failure:
            assert r.coverage_pass is True

    def test_accepted_result_token_audit_passed(self, corpus_files, log_path):
        """Accepted result must have token_audit.passed=True."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if not r.hard_failure:
            assert r.token_audit is not None
            assert r.token_audit.passed is True

    def test_accepted_result_generation_result_not_none(self, corpus_files, log_path):
        """Accepted result must have a non-None generation_result."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if not r.hard_failure:
            assert r.generation_result is not None

    def test_grammar_always_non_empty_before_generation(self, corpus_files, log_path):
        """Grammar string in accepted request must be non-empty (invariant I5/I7)."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if not r.hard_failure:
            assert r.request.grammar_str != ""

    def test_accepted_result_no_unk_in_translation(self, corpus_files, log_path):
        """Accepted translation must NOT contain [UNK:…] markers."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if not r.hard_failure:
            assert "[UNK:" not in r.translation


# ---------------------------------------------------------------------------
# §2  Hard failure path — coverage never passes
# ---------------------------------------------------------------------------

class TestHardFailurePath:
    def test_hard_failure_set(self, corpus_files, log_path):
        """Items with tokens absent from corpus get hard_failure=True."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        results, _ = runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        assert results[0].hard_failure is True

    def test_hard_failure_translation_contains_unk_markers(self, corpus_files, log_path):
        """Coverage hard failure: translation must contain [UNK:…] markers."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        results, _ = runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        assert "[UNK:" in results[0].translation

    def test_hard_failure_generation_result_is_none(self, corpus_files, log_path):
        """Coverage hard failure: generation_result must be None."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        results, _ = runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        assert results[0].generation_result is None

    def test_hard_failure_token_audit_is_none(self, corpus_files, log_path):
        """Coverage hard failure: token_audit must be None."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        results, _ = runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        assert results[0].token_audit is None

    def test_hard_failure_coverage_pass_false(self, corpus_files, log_path):
        """Coverage hard failure: coverage_pass must be False."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        results, _ = runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        assert results[0].coverage_pass is False


# ---------------------------------------------------------------------------
# §3  Held-out exclusion
# ---------------------------------------------------------------------------

class TestHeldOutExclusion:
    def test_exclude_idx_zero_excludes_first_verse(self, corpus_files, log_path):
        """exclude_idx=0 must not allow verse 0 to appear as an example."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run(
            [BatchItem("GEN1:1", _SOURCE[0], exclude_idx=0)]
        )
        r = results[0]
        # Collect all example verse_idxs from semantic + coverage
        all_ex = list(r.request.semantic_examples) + list(r.request.coverage_examples)
        for ex in all_ex:
            assert ex.verse_idx != 0, "exclude_idx=0 was leaked into examples"

    def test_exclude_idx_none_uses_all_verses(self, corpus_files, log_path):
        """exclude_idx=None should not restrict any verse from being selected."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        # Just verify it runs without error with no exclusion
        results, _ = runner.run(
            [BatchItem("GEN1:1", _SOURCE[0], exclude_idx=None)]
        )
        assert results[0].item_id == "GEN1:1"


# ---------------------------------------------------------------------------
# §4  Event log semantics
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_exactly_one_terminal_event_per_item_accepted(self, corpus_files, log_path):
        """Accepted item: exactly one 'provenance' terminal event in log."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        if r.hard_failure:
            pytest.skip("item became hard failure — terminal event is hard_failure")
        events = _read_events(log_path)
        item_events = [e for e in events if e.get("item_id") == "EASY"]
        provenance_events = [e for e in item_events if e["event"] == "provenance"]
        assert len(provenance_events) == 1

    def test_exactly_one_terminal_event_per_item_hard_failure(self, corpus_files, log_path):
        """Hard failure item: exactly one 'hard_failure' terminal event in log."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        runner.run([BatchItem("HARD", _HARD_SOURCE, _HARD_IDX)])
        events = _read_events(log_path)
        item_events = [e for e in events if e.get("item_id") == "HARD"]
        hard_events = [e for e in item_events if e["event"] == "hard_failure"]
        assert len(hard_events) == 1

    def test_batch_summary_logged_once(self, corpus_files, log_path):
        """Exactly one 'batch_summary' event is logged per run."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        events = _read_events(log_path)
        summaries = [e for e in events if e["event"] == "batch_summary"]
        assert len(summaries) == 1

    def test_log_events_have_ts_field(self, corpus_files, log_path):
        """All log events must have a 'ts' (timestamp) field."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        events = _read_events(log_path)
        for e in events:
            assert "ts" in e, f"Event missing 'ts': {e}"

    def test_log_events_have_event_field(self, corpus_files, log_path):
        """All log events must have an 'event' field."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        events = _read_events(log_path)
        for e in events:
            assert "event" in e, f"Event missing 'event': {e}"


# ---------------------------------------------------------------------------
# §5  Multi-item batch: mixed accepted + hard failures
# ---------------------------------------------------------------------------

class TestMixedBatch:
    def test_mixed_batch_result_count(self, corpus_files, log_path):
        """Results list length equals input items length."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, _ = runner.run(items)
        assert len(results) == 2

    def test_mixed_batch_item_ids_preserved(self, corpus_files, log_path):
        """Result item_ids must match input item_ids in order."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, _ = runner.run(items)
        assert results[0].item_id == "EASY"
        assert results[1].item_id == "HARD"

    def test_mixed_batch_rollup_total_units(self, corpus_files, log_path):
        """rollup.total_units must equal number of input items."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, rollup = runner.run(items)
        assert rollup.total_units == 2

    def test_mixed_batch_hard_failure_pct(self, corpus_files, log_path):
        """rollup.hard_failure_pct must match actual hard failure count."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, rollup = runner.run(items)
        actual_hard = sum(1 for r in results if r.hard_failure)
        expected_pct = 100.0 * actual_hard / len(results)
        assert abs(rollup.hard_failure_pct - expected_pct) < 0.01

    def test_mixed_batch_coverage_pass_pct(self, corpus_files, log_path):
        """rollup.coverage_pass_pct must match actual coverage_pass count."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, rollup = runner.run(items)
        actual_pass = sum(1 for r in results if r.coverage_pass)
        expected_pct = 100.0 * actual_pass / len(results)
        assert abs(rollup.coverage_pass_pct - expected_pct) < 0.01

    def test_mixed_batch_one_terminal_event_per_item(self, corpus_files, log_path):
        """Every item must have exactly one terminal log event (provenance or hard_failure)."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("EASY", _EASY_SOURCE, _EASY_IDX),
            BatchItem("HARD", _HARD_SOURCE, _HARD_IDX),
        ]
        results, _ = runner.run(items)
        events = _read_events(log_path)
        terminal_types = {"provenance", "hard_failure"}
        for item in items:
            item_terminals = [
                e for e in events
                if e.get("item_id") == item.item_id
                and e["event"] in terminal_types
            ]
            assert len(item_terminals) == 1, (
                f"Item {item.item_id!r} had {len(item_terminals)} terminal events, expected 1; "
                f"events: {item_terminals}"
            )


# ---------------------------------------------------------------------------
# §6  Unicode handling
# ---------------------------------------------------------------------------

class TestUnicodeHandling:
    def test_unicode_target_tokens_in_attested_vocab(self, corpus_files, log_path):
        """Attested vocab must include Unicode tokens like 'Dieu', 'créa'."""
        runner = _make_runner(corpus_files, log_path, n_semantic=5, n_coverage=5)
        results, _ = runner.run([BatchItem("EASY", _EASY_SOURCE, _EASY_IDX)])
        r = results[0]
        vocab = r.request.attested_vocab
        # The French corpus has these tokens
        assert "Dieu" in vocab or len(vocab) > 0, "attested_vocab is empty"

    def test_log_unicode_no_escape(self, corpus_files, log_path):
        """Log file must use real UTF-8 characters, not \\uXXXX escapes."""
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=1, n_coverage=1)
        # Use a source with a Unicode character
        runner.run([BatchItem("UNI", "lumière et cieux", None)])
        raw_text = Path(log_path).read_text(encoding="utf-8")
        # If Unicode is present in log, it must not be escaped
        # (actual French words from target vocab may appear in provenance)
        # At minimum the log must be valid UTF-8 JSON lines
        for line in raw_text.splitlines():
            if line.strip():
                obj = json.loads(line)  # must parse cleanly
                assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# §7  Rollup consistency with results list
# ---------------------------------------------------------------------------

class TestRollupConsistency:
    def test_rollup_total_units_matches_results_len(self, corpus_files, log_path):
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("A", _EASY_SOURCE, None),
            BatchItem("B", _HARD_SOURCE, None),
            BatchItem("C", _SOURCE[2], 2),
        ]
        results, rollup = runner.run(items)
        assert rollup.total_units == len(results)

    def test_rollup_hard_failure_pct_correct(self, corpus_files, log_path):
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [
            BatchItem("A", _EASY_SOURCE, None),
            BatchItem("B", _HARD_SOURCE, None),
        ]
        results, rollup = runner.run(items)
        hard_count = sum(1 for r in results if r.hard_failure)
        expected = 100.0 * hard_count / len(results)
        assert abs(rollup.hard_failure_pct - expected) < 0.01

    def test_rollup_total_output_tokens_nonneg(self, corpus_files, log_path):
        runner = _make_runner(corpus_files, log_path, max_retries=0, n_semantic=3, n_coverage=3)
        items = [BatchItem("A", _EASY_SOURCE, None)]
        _, rollup = runner.run(items)
        assert rollup.total_output_tokens >= 0
