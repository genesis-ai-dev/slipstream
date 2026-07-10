"""constrained_translation.batch_runner — Coverage-first constrained batch loop.

Task 9 implementation.

Architecture
────────────
For each BatchItem the runner executes this pipeline:

1. **Preflight** — assert backend.is_available(); raise RuntimeError loudly if not.

2. **Attempt 0** — select n_semantic + n_coverage examples via ExampleSelector,
   passing the item's exclude_idx on every call (held-out invariant).

3. **Source coverage check** — build a source-derived vocabulary from the
   *source* strings of the selected examples (NOT from target attested_vocab),
   then call UNKDetector.detect(source_text, source_vocab).  If the result is
   empty → coverage passes.

4. **Coverage retry loop** — if coverage gaps remain after attempt 0:
   a. Log ``coverage_failure`` with uncovered span details.
   b. Retry by expanding coverage-targeted retrieval (increase n_coverage by
      RETRY_COVERAGE_INCREMENT per retry).  exclude_idx is always forwarded.
   c. Log ``coverage_retry`` with old/new example IDs and retry count.
   d. Re-check coverage.
   e. Repeat up to max_retries times.
   f. If coverage still fails after max_retries: hard-fail.  Log
      ``hard_failure``.  Build deterministic [UNK:surface] artifact.
      Set generation_result=None, token_audit=None.  Return immediately.

5. **Target vocab + grammar** — extract target attested_vocab via
   VocabExtractor; build grammar via GrammarBuilder.  GrammarBuildError →
   hard failure logged and returned; generate() is never called.

6. **Generate** — call backend.generate(prompt, grammar, …).  Grammar must be
   non-empty (invariant I7).  Any exception (BackendError or other) → log
   ``hard_failure(reason="backend_generation_failure")``, return immediately,
   NO retry, continue remaining items.

7. **Token audit** — call TokenAuditor.audit().  If audit fails:
   log ``token_audit_violation`` then ``hard_failure``
   (translation = empty failure artifact, NOT the model text).

8. **Accept** — result is accepted only when coverage_pass=True AND
   token_audit.passed=True.  Log ``provenance`` (additional) then ``success``
   (terminal).

9. **Batch summary** — after all items, compute BatchRollup via RollupStats,
   log ``batch_summary``, return (results, rollup).

Hard-behaviour invariants
────────────────────────
I1: Source coverage vocab is built from selected EXAMPLE SOURCE strings.
I2: Target attested_vocab is never passed to UNKDetector for coverage checking.
I3: exclude_idx is forwarded to ExampleSelector on every call (including retries).
I4: backend.generate is never called if coverage has not passed.
I5: backend.generate is never called if grammar is empty or build failed.
I6: Token audit failure → translation is empty (not the model text).
I7: Accepted result only when coverage_pass=True AND token_audit.passed=True.
I8: Backend generation failure (any exception) → hard_failure, no retry,
    remaining items continue unaffected.
I9: Exactly one terminal event {success, hard_failure} per item in the log.
    provenance is additional (not a terminal).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from constrained_translation.example_selector import ExampleSelector
from constrained_translation.grammar_builder import GrammarBuildError, GrammarBuilder
from constrained_translation.logger import JSONLLogger
from constrained_translation.prompt_builder import PromptBuilder
from constrained_translation.protocol import (
    AlignedExample,
    BackendProtocol,
    BatchRollup,
    GenerationResult,
    TokenAuditResult,
    TranslationRequest,
    TranslationResult,
    UNKSpan,
)
from constrained_translation.rollup import RollupStats
from constrained_translation.token_auditor import TokenAuditor, TokenLicense
from constrained_translation.unk_detector import UNKDetector
from constrained_translation.vocab_extractor import VocabExtractor
from constrained_translation.vllm_backend import BackendError


# ---------------------------------------------------------------------------
# BatchItem — explicit per-item held-out specification
# ---------------------------------------------------------------------------

@dataclass
class BatchItem:
    """Input record for one translation unit in a batch run.

    Attributes
    ----------
    item_id:
        Unique string identifier for this translation unit (e.g. "GEN 1:1").
    source_text:
        The source-language text to translate.
    exclude_idx:
        0-based corpus index of the held-out verse for this item.  Passed to
        every ExampleSelector.select() call to prevent leakage.  Pass None
        to include all verses (no held-out exclusion).
    """
    item_id: str
    source_text: str
    exclude_idx: Optional[int]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How much to increment n_coverage on each retry
_RETRY_COVERAGE_INCREMENT = 2

# Default generation parameters
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.0

# Failure artifact suffix
_COVERAGE_FAILURE_ARTIFACT_SUFFIX = " [COVERAGE_FAILURE]"
_AUDIT_FAILURE_ARTIFACT = "[AUDIT_FAILURE]"


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Coverage-first constrained translation batch loop.

    Parameters
    ----------
    source_file:
        Path to the aligned source corpus (one verse per line).
    target_file:
        Path to the aligned target corpus (same order as source_file).
    backend:
        Backend implementing BackendProtocol (FakeBackend or VLLMBackend).
    log_path:
        Destination JSONL log file (appended if it already exists).
    max_retries:
        Maximum number of coverage-expansion retries after the initial attempt.
        0 means only one attempt (no retries).
    n_semantic:
        Number of semantic (BM25) examples to retrieve per attempt.
    n_coverage:
        Base number of coverage-targeted examples to retrieve on attempt 0.
        Expanded by _RETRY_COVERAGE_INCREMENT on each retry.
    max_tokens:
        Maximum tokens to generate per item.
    temperature:
        Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        source_file: str,
        target_file: str,
        backend: BackendProtocol,
        log_path: str,
        max_retries: int = 2,
        n_semantic: int = 5,
        n_coverage: int = 5,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._source_file = source_file
        self._target_file = target_file
        self._backend = backend
        self._log_path = log_path
        self._max_retries = max_retries
        self._n_semantic = n_semantic
        self._n_coverage = n_coverage
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Sub-components (constructed fresh; no shared state between run() calls)
        self._vocab_extractor = VocabExtractor()
        self._grammar_builder = GrammarBuilder()
        self._prompt_builder = PromptBuilder()
        self._unk_detector = UNKDetector()
        self._token_auditor = TokenAuditor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        items: list[BatchItem],
    ) -> tuple[list[TranslationResult], BatchRollup]:
        """Translate a batch of items with coverage-first constrained generation.

        Parameters
        ----------
        items:
            List of BatchItem records, each with its own held-out exclude_idx.

        Returns
        -------
        (list[TranslationResult], BatchRollup)
            Per-item results and batch-level aggregate statistics.

        Raises
        ------
        RuntimeError
            If the backend reports is_available() == False (loud preflight).
        """
        # ── Preflight ────────────────────────────────────────────────────────
        if not self._backend.is_available():
            raise RuntimeError(
                "Backend is unavailable — cannot run batch translation. "
                "Check that the backend is reachable/ready before calling run()."
            )

        results: list[TranslationResult] = []

        with JSONLLogger(self._log_path) as logger:
            # Build ExampleSelector once (loads BM25 index)
            selector = ExampleSelector(
                source_file=self._source_file,
                target_file=self._target_file,
                n_semantic=self._n_semantic,
                n_coverage=self._n_coverage,
            )

            for item in items:
                result = self._process_item(item, selector, logger)
                results.append(result)

            # ── Batch summary ─────────────────────────────────────────────
            rollup = RollupStats.compute(results)
            logger.log(
                "batch_summary",
                item_id="batch",
                total_units=rollup.total_units,
                coverage_pass_pct=rollup.coverage_pass_pct,
                retry_pct=rollup.retry_pct,
                hard_failure_pct=rollup.hard_failure_pct,
                token_audit_pass_pct=rollup.token_audit_pass_pct,
                mean_tok_per_sec=rollup.mean_tok_per_sec,
                total_input_tokens=rollup.total_input_tokens,
                total_output_tokens=rollup.total_output_tokens,
            )

        return results, rollup

    # ------------------------------------------------------------------
    # Private: per-item pipeline
    # ------------------------------------------------------------------

    def _process_item(
        self,
        item: BatchItem,
        selector: ExampleSelector,
        logger: JSONLLogger,
    ) -> TranslationResult:
        """Run the full pipeline for one BatchItem."""

        item_id = item.item_id
        source_text = item.source_text
        exclude_idx = item.exclude_idx

        # ── Attempt 0: initial example selection ─────────────────────────
        examples = selector.select(source_text, exclude_idx=exclude_idx)
        retry_count = 0

        # ── Coverage check + retry loop ───────────────────────────────────
        unk_spans = self._check_source_coverage(source_text, examples)
        coverage_pass = len(unk_spans) == 0

        if not coverage_pass:
            # Log initial coverage failure
            logger.log(
                "coverage_failure",
                item_id=item_id,
                uncovered_spans=[s.surface for s in unk_spans],
                uncovered_count=len(unk_spans),
                attempt=0,
                example_ids=[ex.verse_idx for ex in examples],
            )

            # Retry loop: expand coverage-targeted retrieval
            for retry in range(1, self._max_retries + 1):
                old_example_ids = [ex.verse_idx for ex in examples]

                # Rebuild selector with expanded n_coverage
                expanded_n_coverage = self._n_coverage + retry * _RETRY_COVERAGE_INCREMENT
                retry_selector = ExampleSelector(
                    source_file=self._source_file,
                    target_file=self._target_file,
                    n_semantic=self._n_semantic,
                    n_coverage=expanded_n_coverage,
                )
                examples = retry_selector.select(source_text, exclude_idx=exclude_idx)
                new_example_ids = [ex.verse_idx for ex in examples]

                # Re-check coverage
                unk_spans = self._check_source_coverage(source_text, examples)
                coverage_pass = len(unk_spans) == 0

                logger.log(
                    "coverage_retry",
                    item_id=item_id,
                    retry_count=retry,
                    old_example_ids=old_example_ids,
                    new_example_ids=new_example_ids,
                    uncovered_spans=[s.surface for s in unk_spans],
                    coverage_pass=coverage_pass,
                )

                if coverage_pass:
                    break

            retry_count = self._max_retries if not coverage_pass else retry

        # ── Hard failure: coverage never passed ───────────────────────────
        if not coverage_pass:
            # Build deterministic [UNK:surface] failure artifact
            artifact = self._build_unk_artifact(source_text, unk_spans)

            logger.log(
                "hard_failure",
                item_id=item_id,
                reason="coverage_failure",
                uncovered_spans=[s.surface for s in unk_spans],
                retry_count=retry_count,
                artifact=artifact,
            )

            # Build a minimal stub TranslationRequest for the result
            stub_request = TranslationRequest(
                item_id=item_id,
                source_text=source_text,
                semantic_examples=tuple(
                    ex for ex in examples if ex.selection_method == "semantic"
                ),
                coverage_examples=tuple(
                    ex for ex in examples if ex.selection_method == "coverage"
                ),
                attested_vocab=frozenset(),
                unk_spans=unk_spans,
                grammar_str="",
                prompt="",
            )

            return TranslationResult(
                item_id=item_id,
                source_text=source_text,
                translation=artifact,
                coverage_pass=False,
                retry_count=retry_count,
                hard_failure=True,
                generation_result=None,
                token_audit=None,
                request=stub_request,
                error=f"Coverage failure after {retry_count} retries; "
                      f"{len(unk_spans)} uncovered span(s): "
                      + ", ".join(s.surface for s in unk_spans),
            )

        # ── Coverage passed — extract target vocab ────────────────────────
        attested_vocab = self._vocab_extractor.extract(examples)

        # ── Build grammar (failure → hard failure, no generate) ───────────
        try:
            grammar_str = self._grammar_builder.build(attested_vocab, item_id)
        except GrammarBuildError as exc:
            logger.log(
                "hard_failure",
                item_id=item_id,
                reason="grammar_build_error",
                error=str(exc),
                retry_count=retry_count,
            )

            stub_request = TranslationRequest(
                item_id=item_id,
                source_text=source_text,
                semantic_examples=tuple(
                    ex for ex in examples if ex.selection_method == "semantic"
                ),
                coverage_examples=tuple(
                    ex for ex in examples if ex.selection_method == "coverage"
                ),
                attested_vocab=attested_vocab,
                unk_spans=unk_spans,
                grammar_str="",
                prompt="",
            )

            return TranslationResult(
                item_id=item_id,
                source_text=source_text,
                translation="",
                coverage_pass=True,
                retry_count=retry_count,
                hard_failure=True,
                generation_result=None,
                token_audit=None,
                request=stub_request,
                error=str(exc),
            )

        # ── Assemble prompt ───────────────────────────────────────────────
        request = TranslationRequest(
            item_id=item_id,
            source_text=source_text,
            semantic_examples=tuple(
                ex for ex in examples if ex.selection_method == "semantic"
            ),
            coverage_examples=tuple(
                ex for ex in examples if ex.selection_method == "coverage"
            ),
            attested_vocab=attested_vocab,
            unk_spans=unk_spans,   # empty (coverage passed)
            grammar_str=grammar_str,
            prompt="",  # filled below
        )
        prompt = self._prompt_builder.build(request)
        request.prompt = prompt

        # ── Generate (grammar must be non-empty — invariant I5) ───────────
        assert grammar_str, "Grammar must be non-empty before calling generate()"
        try:
            gen_result: GenerationResult = self._backend.generate(
                prompt=prompt,
                grammar=grammar_str,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:  # noqa: BLE001 — BackendError or any backend failure
            logger.log(
                "hard_failure",
                item_id=item_id,
                reason="backend_generation_failure",
                error=str(exc),
                retry_count=retry_count,
            )
            return TranslationResult(
                item_id=item_id,
                source_text=source_text,
                translation="",
                coverage_pass=True,
                retry_count=retry_count,
                hard_failure=True,
                generation_result=None,
                token_audit=None,
                request=request,
                error=f"Backend generation failed: {exc}",
            )

        # ── Token audit ───────────────────────────────────────────────────
        all_examples = list(examples)
        audit_result, provenance_map = self._token_auditor.audit(
            token_ids=gen_result.token_ids,
            attested_vocab=attested_vocab,
            backend=self._backend,
            examples=all_examples,
        )

        if not audit_result.passed:
            # Token audit failed → log violation event, then hard failure; reject model text
            logger.log(
                "token_audit_violation",
                item_id=item_id,
                violations=audit_result.violations,
                retry_count=retry_count,
            )
            logger.log(
                "hard_failure",
                item_id=item_id,
                reason="token_audit_failure",
                violations=audit_result.violations,
                retry_count=retry_count,
            )

            return TranslationResult(
                item_id=item_id,
                source_text=source_text,
                translation=_AUDIT_FAILURE_ARTIFACT,
                coverage_pass=True,
                retry_count=retry_count,
                hard_failure=True,
                generation_result=gen_result,
                token_audit=audit_result,
                request=request,
                error="Token audit failed: " + "; ".join(audit_result.violations[:3]),
            )

        # ── Accepted — log provenance and success ────────────────────────
        # Convert int keys to strings for JSON serialization
        json_provenance = {str(k): v for k, v in provenance_map.items()}
        logger.log(
            "provenance",
            item_id=item_id,
            token_to_verse=json_provenance,
            retry_count=retry_count,
        )
        logger.log(
            "success",
            item_id=item_id,
            retry_count=retry_count,
        )

        return TranslationResult(
            item_id=item_id,
            source_text=source_text,
            translation=gen_result.text,
            coverage_pass=True,
            retry_count=retry_count,
            hard_failure=False,
            generation_result=gen_result,
            token_audit=audit_result,
            request=request,
            error=None,
        )

    # ------------------------------------------------------------------
    # Private: coverage helpers
    # ------------------------------------------------------------------

    def _check_source_coverage(
        self,
        source_text: str,
        examples: tuple[AlignedExample, ...],
    ) -> tuple[UNKSpan, ...]:
        """Check source coverage using SOURCE-SIDE example strings.

        Builds a vocabulary from the *source* strings of the selected examples
        (NOT from the target attested_vocab) and passes it to UNKDetector.
        This ensures invariant I1: coverage is never derived from target vocab.

        Returns
        -------
        tuple[UNKSpan, ...]
            Uncovered source spans.  Empty tuple means full coverage.
        """
        # Build source-derived vocabulary from example SOURCE strings.
        source_vocab: set[str] = set()
        for ex in examples:
            for tok in ex.source.split():
                nfkc = unicodedata.normalize("NFKC", tok)
                source_vocab.add(nfkc)

        return self._unk_detector.detect(source_text, frozenset(source_vocab))

    # ------------------------------------------------------------------
    # Private: failure artifact builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_unk_artifact(
        source_text: str,
        unk_spans: tuple[UNKSpan, ...],
    ) -> str:
        """Build a deterministic failure artifact with [UNK:surface] markers.

        Substitutions are applied right-to-left (descending start_char) so
        that earlier offsets remain valid after each replacement.

        Parameters
        ----------
        source_text:
            Original source sentence.
        unk_spans:
            UNKSpans to mark.  May be empty (returns source_text unchanged).

        Returns
        -------
        str
            Source text with each uncovered span replaced by [UNK:surface].
        """
        if not unk_spans:
            return source_text

        result = source_text
        for span in sorted(unk_spans, key=lambda s: s.start_char, reverse=True):
            marker = f"[UNK:{span.surface}]"
            result = result[: span.start_char] + marker + result[span.end_char :]

        return result
