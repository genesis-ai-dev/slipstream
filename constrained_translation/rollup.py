"""constrained_translation.rollup — RollupStats aggregate metrics.

Computes batch-level statistics from a list of TranslationResult records.

Zero-denominator policy (well-defined behaviour):
  - Any percentage with an empty denominator returns 0.0.
  - tok/s with no accepted non-hard-failure generations returns 0.0.
  - token_audit_pass_pct denominator is items with a non-None TokenAuditResult
    (items that were never audited are excluded from the denominator entirely).
"""
from __future__ import annotations

from typing import List

from .protocol import BatchRollup, TranslationResult


class RollupStats:
    """Compute aggregate metrics across a batch of TranslationResult records.

    Usage::

        rollup = RollupStats.compute(results)
    """

    @staticmethod
    def compute(results: List[TranslationResult]) -> BatchRollup:
        """Aggregate *results* into a BatchRollup.

        Parameters
        ----------
        results:
            List of TranslationResult objects produced by the batch runner.
            May be empty (returns all-zero BatchRollup).

        Returns
        -------
        BatchRollup
            Aggregate statistics with well-defined behaviour for zero
            denominators.

        tok/s definition:
            ``total_output_tokens / (total_generation_ms / 1000)`` summed over
            all *accepted* (non-hard-failure) results that have a non-None
            GenerationResult.  Hard-failure results are excluded because they
            represent rejected/aborted generations, not accepted output.
        """
        total_units = len(results)

        if total_units == 0:
            return BatchRollup(
                total_units=0,
                mean_tok_per_sec=0.0,
                coverage_pass_pct=0.0,
                retry_pct=0.0,
                hard_failure_pct=0.0,
                token_audit_pass_pct=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
            )

        # --- percentage metrics with denominator = total_units ---------------
        coverage_pass_count = sum(1 for r in results if r.coverage_pass)
        retry_count = sum(1 for r in results if r.retry_count > 0)
        hard_failure_count = sum(1 for r in results if r.hard_failure)

        coverage_pass_pct = 100.0 * coverage_pass_count / total_units
        retry_pct = 100.0 * retry_count / total_units
        hard_failure_pct = 100.0 * hard_failure_count / total_units

        # --- token audit percentage (denominator = items with audit result) --
        audited = [r for r in results if r.token_audit is not None]
        if audited:
            audit_pass_count = sum(1 for r in audited if r.token_audit.passed)
            token_audit_pass_pct = 100.0 * audit_pass_count / len(audited)
        else:
            token_audit_pass_pct = 0.0

        # --- token totals and tok/s ------------------------------------------
        # Only accepted (non-hard-failure) results with a GenerationResult
        # contribute to tok/s and token totals.
        accepted = [
            r for r in results
            if not r.hard_failure and r.generation_result is not None
        ]

        total_output_tokens = sum(r.generation_result.output_tokens for r in accepted)
        total_generation_ms = sum(r.generation_result.generation_ms for r in accepted)

        if total_generation_ms > 0.0:
            mean_tok_per_sec = total_output_tokens / (total_generation_ms / 1000.0)
        else:
            mean_tok_per_sec = 0.0

        # Input tokens: sum over all results with a GenerationResult
        # (including hard failures — they may have consumed prompt tokens).
        all_with_gen = [r for r in results if r.generation_result is not None]
        total_input_tokens = sum(r.generation_result.prompt_tokens for r in all_with_gen)
        # total_output_tokens already computed from accepted; add hard-failure
        # output tokens separately so the field reflects actual generated tokens.
        # Per plan §5.7: total_output_tokens is a raw sum across all results.
        total_output_tokens_all = sum(
            r.generation_result.output_tokens for r in all_with_gen
        )

        return BatchRollup(
            total_units=total_units,
            mean_tok_per_sec=mean_tok_per_sec,
            coverage_pass_pct=coverage_pass_pct,
            retry_pct=retry_pct,
            hard_failure_pct=hard_failure_pct,
            token_audit_pass_pct=token_audit_pass_pct,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens_all,
        )
