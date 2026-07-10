"""Tests for RollupStats (Task 11).

TDD — these tests are written *before* the implementation.
They must fail with ImportError or AttributeError first, then pass once
rollup.py is implemented.

Coverage:
    test_rollup_total_units
    test_rollup_coverage_pass_pct
    test_rollup_retry_pct
    test_rollup_hard_failure_pct
    test_rollup_tok_per_sec_excludes_hard_failures
    test_rollup_token_audit_pass_pct
    test_rollup_all_zeros_on_empty_results
    test_rollup_tok_per_sec_zero_when_no_accepted_generations
    test_rollup_input_and_output_token_totals
    test_rollup_token_audit_pass_pct_ignores_none_audits
"""
from __future__ import annotations

import pytest

from constrained_translation.rollup import RollupStats
from constrained_translation.protocol import (
    AlignedExample,
    BatchRollup,
    GenerationResult,
    TokenAuditResult,
    TranslationRequest,
    TranslationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(item_id: str = "TEST 1:1") -> TranslationRequest:
    """Minimal TranslationRequest for use in TranslationResult fixtures."""
    return TranslationRequest(
        item_id=item_id,
        source_text="Hello world",
        semantic_examples=(),
        coverage_examples=(),
        attested_vocab=frozenset(["Hello", "world"]),
        unk_spans=(),
        grammar_str='root ::= "Hello" | "world"',
        prompt="[Source]: Hello world\n[Translation]:",
    )


def _gen(output_tokens: int = 10, generation_ms: float = 500.0,
         prompt_tokens: int = 5) -> GenerationResult:
    """Factory for a minimal GenerationResult."""
    return GenerationResult(
        text="hello world",
        token_ids=list(range(output_tokens)),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        generation_ms=generation_ms,
    )


def _audit(passed: bool = True) -> TokenAuditResult:
    return TokenAuditResult(passed=passed, violations=[])


def _result(
    item_id: str = "TEST 1:1",
    coverage_pass: bool = True,
    retry_count: int = 0,
    hard_failure: bool = False,
    generation_result: GenerationResult | None = None,
    token_audit: TokenAuditResult | None = None,
    error: str | None = None,
) -> TranslationResult:
    return TranslationResult(
        item_id=item_id,
        source_text="Hello world",
        translation="hello world",
        coverage_pass=coverage_pass,
        retry_count=retry_count,
        hard_failure=hard_failure,
        generation_result=generation_result,
        token_audit=token_audit,
        request=_make_request(item_id),
        error=error,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRollupTotalUnits:
    def test_rollup_total_units_single(self):
        results = [_result()]
        rollup = RollupStats.compute(results)
        assert rollup.total_units == 1

    def test_rollup_total_units_multiple(self):
        results = [_result(item_id=f"TEST {i}:1") for i in range(5)]
        rollup = RollupStats.compute(results)
        assert rollup.total_units == 5

    def test_rollup_total_units_empty(self):
        rollup = RollupStats.compute([])
        assert rollup.total_units == 0


class TestRollupCoveragePassPct:
    def test_rollup_coverage_pass_pct_all_pass(self):
        results = [_result(coverage_pass=True) for _ in range(4)]
        rollup = RollupStats.compute(results)
        assert rollup.coverage_pass_pct == pytest.approx(100.0)

    def test_rollup_coverage_pass_pct_none_pass(self):
        results = [_result(coverage_pass=False) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.coverage_pass_pct == pytest.approx(0.0)

    def test_rollup_coverage_pass_pct_mixed(self):
        results = [
            _result(coverage_pass=True),
            _result(coverage_pass=True),
            _result(coverage_pass=False),
            _result(coverage_pass=False),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.coverage_pass_pct == pytest.approx(50.0)


class TestRollupRetryPct:
    def test_rollup_retry_pct_no_retries(self):
        results = [_result(retry_count=0) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.retry_pct == pytest.approx(0.0)

    def test_rollup_retry_pct_all_retried(self):
        results = [_result(retry_count=1) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.retry_pct == pytest.approx(100.0)

    def test_rollup_retry_pct_mixed(self):
        results = [
            _result(retry_count=0),
            _result(retry_count=1),
            _result(retry_count=2),
            _result(retry_count=0),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.retry_pct == pytest.approx(50.0)  # 2/4 have retry_count > 0


class TestRollupHardFailurePct:
    def test_rollup_hard_failure_pct_none(self):
        results = [_result(hard_failure=False) for _ in range(4)]
        rollup = RollupStats.compute(results)
        assert rollup.hard_failure_pct == pytest.approx(0.0)

    def test_rollup_hard_failure_pct_all(self):
        results = [_result(hard_failure=True) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.hard_failure_pct == pytest.approx(100.0)

    def test_rollup_hard_failure_pct_one_of_four(self):
        results = [
            _result(hard_failure=False),
            _result(hard_failure=False),
            _result(hard_failure=False),
            _result(hard_failure=True),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.hard_failure_pct == pytest.approx(25.0)


class TestRollupTokPerSec:
    def test_rollup_tok_per_sec_excludes_hard_failures(self):
        """Hard-failure results have no accepted generation; exclude them."""
        good = _result(
            item_id="GOOD 1:1",
            hard_failure=False,
            generation_result=_gen(output_tokens=20, generation_ms=1000.0),
        )
        bad = _result(
            item_id="BAD 1:1",
            hard_failure=True,
            generation_result=_gen(output_tokens=50, generation_ms=100.0),
        )
        rollup = RollupStats.compute([good, bad])
        # Only 'good' counts: 20 tokens / 1.0 s = 20 tok/s
        assert rollup.mean_tok_per_sec == pytest.approx(20.0)

    def test_rollup_tok_per_sec_uses_output_tokens_and_gen_ms(self):
        """tok/s = total_output_tokens / (total_generation_ms / 1000)."""
        r1 = _result(
            item_id="R1",
            generation_result=_gen(output_tokens=10, generation_ms=500.0),
        )
        r2 = _result(
            item_id="R2",
            generation_result=_gen(output_tokens=30, generation_ms=1500.0),
        )
        rollup = RollupStats.compute([r1, r2])
        # (10 + 30) / ((500 + 1500) / 1000) = 40 / 2 = 20 tok/s
        assert rollup.mean_tok_per_sec == pytest.approx(20.0)

    def test_rollup_tok_per_sec_zero_when_no_accepted_generations(self):
        """If all results are hard failures (no accepted generations) → 0.0."""
        results = [
            _result(hard_failure=True, generation_result=_gen()),
            _result(hard_failure=True, generation_result=_gen()),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.mean_tok_per_sec == pytest.approx(0.0)

    def test_rollup_tok_per_sec_skips_none_generation_result(self):
        """Results with generation_result=None are excluded from tok/s."""
        r_with = _result(
            item_id="WITH",
            generation_result=_gen(output_tokens=100, generation_ms=2000.0),
        )
        r_none = _result(item_id="NONE", generation_result=None)
        rollup = RollupStats.compute([r_with, r_none])
        # 100 / 2.0 = 50 tok/s
        assert rollup.mean_tok_per_sec == pytest.approx(50.0)


class TestRollupTokenAuditPassPct:
    def test_rollup_token_audit_pass_pct_all_pass(self):
        results = [_result(token_audit=_audit(passed=True)) for _ in range(4)]
        rollup = RollupStats.compute(results)
        assert rollup.token_audit_pass_pct == pytest.approx(100.0)

    def test_rollup_token_audit_pass_pct_none_pass(self):
        results = [_result(token_audit=_audit(passed=False)) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.token_audit_pass_pct == pytest.approx(0.0)

    def test_rollup_token_audit_pass_pct_ignores_none_audits(self):
        """Items with token_audit=None are excluded from audit denominator."""
        results = [
            _result(item_id="A", token_audit=_audit(passed=True)),
            _result(item_id="B", token_audit=_audit(passed=False)),
            _result(item_id="C", token_audit=None),  # excluded from pct
        ]
        rollup = RollupStats.compute(results)
        # 1 pass out of 2 audited items → 50%
        assert rollup.token_audit_pass_pct == pytest.approx(50.0)

    def test_rollup_token_audit_pass_pct_all_none_gives_zero(self):
        """When no items have an audit result, return 0.0 (well-defined zero denominator)."""
        results = [_result(token_audit=None) for _ in range(3)]
        rollup = RollupStats.compute(results)
        assert rollup.token_audit_pass_pct == pytest.approx(0.0)


class TestRollupTokenTotals:
    def test_rollup_input_and_output_token_totals(self):
        results = [
            _result(
                item_id="A",
                generation_result=_gen(output_tokens=10, prompt_tokens=20),
            ),
            _result(
                item_id="B",
                generation_result=_gen(output_tokens=30, prompt_tokens=40),
            ),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.total_output_tokens == 40
        assert rollup.total_input_tokens == 60

    def test_rollup_token_totals_none_generation_skipped(self):
        results = [
            _result(item_id="A", generation_result=_gen(output_tokens=5, prompt_tokens=10)),
            _result(item_id="B", generation_result=None),
        ]
        rollup = RollupStats.compute(results)
        assert rollup.total_output_tokens == 5
        assert rollup.total_input_tokens == 10


class TestRollupAllZerosOnEmpty:
    def test_rollup_all_zeros_on_empty_results(self):
        """Empty input returns a BatchRollup with all numeric fields = 0."""
        rollup = RollupStats.compute([])
        assert isinstance(rollup, BatchRollup)
        assert rollup.total_units == 0
        assert rollup.mean_tok_per_sec == pytest.approx(0.0)
        assert rollup.coverage_pass_pct == pytest.approx(0.0)
        assert rollup.retry_pct == pytest.approx(0.0)
        assert rollup.hard_failure_pct == pytest.approx(0.0)
        assert rollup.token_audit_pass_pct == pytest.approx(0.0)
        assert rollup.total_input_tokens == 0
        assert rollup.total_output_tokens == 0


class TestRollupReturnType:
    def test_rollup_returns_batch_rollup_instance(self):
        rollup = RollupStats.compute([_result()])
        assert isinstance(rollup, BatchRollup)

    def test_rollup_compute_is_static_method(self):
        """RollupStats.compute can be called without instantiation."""
        rollup = RollupStats.compute([])
        assert rollup is not None
