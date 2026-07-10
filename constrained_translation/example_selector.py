"""constrained_translation.example_selector — Deterministic two-stage example selection.

Stage 1 (semantic):
    Uses BM25Query to retrieve the top-n_semantic aligned sentence pairs
    whose source text is most similar to the query.

Stage 2 (coverage):
    Uses ContextQuery to retrieve up to n_coverage additional pairs that
    together maximise lexical coverage of query tokens not yet covered.

De-duplication:
    If a verse_idx appears in both stages, the semantic entry is kept and the
    coverage entry is discarded (per plan §7.3).

Held-out exclusion:
    exclude_idx is the 0-based verse index of the query verse itself.  It is
    excluded from BOTH stages by its stable integer index — never by text
    matching (invariant §2, task 3 spec).

Evidence tier:
    All returned AlignedExample objects carry evidence_tier=1
    (sentence co-occurrence).  This is NOT word alignment; it only means the
    paired sentences are mutual translations.
"""

from __future__ import annotations

from typing import Optional

from constrained_translation.protocol import AlignedExample
from query.bm25query import BM25Query
from query.contextquery import ContextQuery


class ExampleSelector:
    """Select aligned example pairs for a source query in two deterministic stages.

    Parameters
    ----------
    source_file:
        Path to aligned source corpus (one verse per line).
    target_file:
        Path to aligned target corpus (one verse per line, same order).
    n_semantic:
        Number of semantic (BM25) candidates to retrieve.
    n_coverage:
        Number of coverage-targeted candidates to retrieve.
    """

    def __init__(
        self,
        source_file: str,
        target_file: str,
        n_semantic: int = 5,
        n_coverage: int = 5,
    ) -> None:
        self._source_file = source_file
        self._target_file = target_file
        self._n_semantic = n_semantic
        self._n_coverage = n_coverage

        # Instantiate query objects once; they load and preprocess the corpus.
        self._bm25 = BM25Query(source_file, target_file, verbose=False)
        self._ctx = ContextQuery(source_file, target_file, verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        query: str,
        exclude_idx: Optional[int] = None,
    ) -> tuple[AlignedExample, ...]:
        """Return at most n_semantic + n_coverage deduplicated AlignedExamples.

        Parameters
        ----------
        query:
            Source-language text to find examples for.
        exclude_idx:
            0-based corpus index of the held-out / query verse.  This verse is
            excluded from results by its stable integer index (not by text
            match).  Pass None to include all verses.

        Returns
        -------
        tuple[AlignedExample, ...]
            Semantic results first, then non-duplicate coverage results.
            All entries have evidence_tier=1.
        """
        exclude: int = exclude_idx if exclude_idx is not None else -1

        # ----------------------------------------------------------------
        # Stage 1 — semantic retrieval via BM25
        # ----------------------------------------------------------------
        # BM25Query._simple_search returns List[Tuple[1-based_idx, src, tgt, score]]
        raw_semantic = self._bm25._simple_search(
            query, top_k=self._n_semantic, exclude_idx=exclude
        )

        semantic_examples: list[AlignedExample] = []
        seen_verse_indices: set[int] = set()

        for one_based_idx, source, target, score in raw_semantic:
            verse_idx = one_based_idx - 1  # convert to 0-based
            if verse_idx == exclude_idx:
                continue
            if verse_idx in seen_verse_indices:
                continue
            seen_verse_indices.add(verse_idx)
            semantic_examples.append(
                AlignedExample(
                    verse_idx=verse_idx,
                    source=source,
                    target=target,
                    selection_score=float(score),
                    selection_method="semantic",
                    evidence_tier=1,
                )
            )

        # ----------------------------------------------------------------
        # Stage 2 — coverage-targeted retrieval via ContextQuery
        # ----------------------------------------------------------------
        # ContextQuery._branching_search returns List[Tuple[1-based_idx, src, tgt, score]]
        raw_coverage = self._ctx._branching_search(
            query, top_k=self._n_coverage, exclude_idx=exclude
        )

        coverage_examples: list[AlignedExample] = []

        for one_based_idx, source, target, score in raw_coverage:
            verse_idx = one_based_idx - 1  # convert to 0-based
            if verse_idx == exclude_idx:
                continue
            if verse_idx in seen_verse_indices:
                # Already present from semantic stage — deduplicate.
                continue
            seen_verse_indices.add(verse_idx)
            coverage_examples.append(
                AlignedExample(
                    verse_idx=verse_idx,
                    source=source,
                    target=target,
                    selection_score=float(score),
                    selection_method="coverage",
                    evidence_tier=1,
                )
            )

        return tuple(semantic_examples + coverage_examples)
