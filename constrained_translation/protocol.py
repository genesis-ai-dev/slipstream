"""constrained_translation.protocol — Data schemas and BackendProtocol.

All dataclasses are frozen (hashable) where specified by the plan, or
mutable where mutability is required by the plan.

BackendProtocol is decorated with @runtime_checkable so isinstance() checks
work in tests and the batch runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# §5.1 AlignedExample — frozen, hashable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedExample:
    """A single aligned sentence pair retrieved from the eBible corpus.

    Evidence tier: always 1 (sentence co-occurrence).  This means the source
    and target strings are translations of each other; it does NOT mean any
    specific source word maps to any specific target word.  This is NOT word
    alignment.
    """

    verse_idx: int           # 0-based index into aligned corpus
    source: str              # raw source verse text
    target: str              # raw target verse text
    selection_score: float   # BM25 / coverage score
    selection_method: str    # "semantic" | "coverage"
    evidence_tier: int       # always 1 (sentence co-occurrence) — see §6


# ---------------------------------------------------------------------------
# §5.2 UNKSpan — frozen, hashable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UNKSpan:
    """A contiguous run of source tokens not covered by the attested vocabulary.

    Inserted deterministically by UNKDetector *before* model generation
    (invariant I3).  The model never generates [UNK:…] text spontaneously.
    """

    surface: str       # source-side span text (used verbatim in [UNK:surface])
    start_char: int    # character offset in source_text
    end_char: int


# ---------------------------------------------------------------------------
# §5.3 TranslationRequest — mutable
# ---------------------------------------------------------------------------

@dataclass
class TranslationRequest:
    """All inputs required to call the backend for one translation unit."""

    item_id: str
    source_text: str
    semantic_examples: tuple[AlignedExample, ...]    # ≤5 items
    coverage_examples: tuple[AlignedExample, ...]   # ≤5 items
    attested_vocab: frozenset[str]                   # surface tokens from target sides
    unk_spans: tuple[UNKSpan, ...]                   # pre-detected uncovered spans
    grammar_str: str                                  # XGrammar grammar string
    prompt: str                                       # assembled few-shot prompt


# ---------------------------------------------------------------------------
# §5.4 GenerationResult — mutable
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Raw output from the backend for one generation call."""

    text: str
    token_ids: list[int]      # as returned by backend
    prompt_tokens: int
    output_tokens: int
    generation_ms: float


# ---------------------------------------------------------------------------
# §5.5 TokenAuditResult — mutable
# ---------------------------------------------------------------------------

@dataclass
class TokenAuditResult:
    """Post-decode token-ID audit result (invariant I2).

    Scalar diagnostic fields
    ------------------------
    token_provenance_ratio : float, [0, 1]
        licensed / generated non-layout token occurrences.  Non-gating — does
        not affect ``passed``.
    surface_attestation_similarity : float, [0, 1]
        Full-sequence decoded lexical words scored against nearest
        ``attested_vocab`` token by 1-normalised Levenshtein, weighted by
        word character length.  Empty lexical output → 0.0.  Non-gating.
    """

    passed: bool
    violations: list[str]    # list of "token_id=N decoded='x' not in attested_vocab"
    token_provenance_ratio: float = 0.0
    surface_attestation_similarity: float = 0.0


# ---------------------------------------------------------------------------
# §5.6 TranslationResult — mutable
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """Final output for one translation unit after the retry loop."""

    item_id: str
    source_text: str
    translation: str          # final text (may contain [UNK:…] markers)
    coverage_pass: bool
    retry_count: int          # 0 = first attempt succeeded
    hard_failure: bool        # True if all retries exhausted
    generation_result: Optional[GenerationResult]
    token_audit: Optional[TokenAuditResult]
    request: TranslationRequest
    error: Optional[str]      # exception message on hard failure


# ---------------------------------------------------------------------------
# §5.7 BatchRollup — mutable
# ---------------------------------------------------------------------------

@dataclass
class BatchRollup:
    """Aggregate metrics across an entire batch run."""

    total_units: int
    mean_tok_per_sec: float
    coverage_pass_pct: float    # % items where coverage_pass=True
    retry_pct: float            # % items with retry_count > 0
    hard_failure_pct: float     # % items with hard_failure=True
    token_audit_pass_pct: float # % items where audit passed
    total_input_tokens: int
    total_output_tokens: int


# ---------------------------------------------------------------------------
# §7.1 BackendProtocol — runtime-checkable
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendProtocol(Protocol):
    """Narrow protocol satisfied by both VLLMBackend and FakeBackend.

    Invariant I6: offline tests use FakeBackend; no GPU or model download
    required.  Invariant I7: if grammar construction fails, raise
    GrammarBuildError — never silently drop the constraint.
    """

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run constrained generation and return a GenerationResult."""
        ...

    def tokenize(self, text: str) -> list[int]:
        """Split *text* into tokens and return their integer IDs."""
        ...

    def decode_token(self, token_id: int) -> str:
        """Return the surface string for a single token ID."""
        ...

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Return the surface string for a full sequence of token IDs.

        This performs full-sequence detokenisation, which is required for
        BPE tokenisers (e.g. Qwen3.5) and byte-level tokenisers where
        decoding individual tokens in isolation produces incorrect or
        garbled output (e.g. U+FFFD byte fragments for multi-byte Unicode
        characters).

        The result should match what the tokeniser's detokenise() / decode()
        method returns for the complete sequence — not a simple concatenation
        of individual decode_token() results.
        """
        ...

    def is_available(self) -> bool:
        """Return True if the backend is reachable / ready."""
        ...
