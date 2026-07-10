"""constrained_translation.example_selector — Deterministic two-stage example selection.

Stage 1 (semantic):
    Uses BM25Query to retrieve the top-n_semantic aligned sentence pairs
    whose source text is most similar to the query.

Stage 2 (coverage):
    Deterministic greedy set-cover over corpus rows not already selected by
    Stage 1 (and not held out).  Normalised query units are computed; any
    units already present in the Stage-1 source sides are subtracted.  The
    algorithm then repeatedly picks the eligible row that covers the most
    currently-uncovered units, breaking ties by BM25 score on the remaining
    uncovered units (higher is better) then by lowest verse index.  Selection
    stops when no gain is possible, all units are covered, or n_coverage
    examples have been added.

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
        # Stage 2 — greedy set-cover over uncovered query units
        # ----------------------------------------------------------------
        # Compute normalised query units (tokens after BM25 normalisation).
        query_units: set[str] = set(
            self._bm25._normalize_text(query).split()
        )

        # Subtract units already covered by semantic stage results.
        covered_units: set[str] = set()
        for ex in semantic_examples:
            covered_units |= (
                set(self._bm25._normalize_text(ex.source).split()) & query_units
            )

        remaining_units = query_units - covered_units

        coverage_examples: list[AlignedExample] = []

        while remaining_units and len(coverage_examples) < self._n_coverage:
            best_idx: int | None = None
            best_gain: int = 0
            best_score: float = -1.0

            for idx in self._bm25.valid_indices:
                if idx == exclude_idx:
                    continue
                if idx in seen_verse_indices:
                    continue

                doc_tokens = set(
                    self._bm25._normalize_text(
                        self._bm25.source_verses[idx]
                    ).split()
                )
                gain = len(doc_tokens & remaining_units)
                if gain == 0:
                    continue

                # BM25 score on only the remaining uncovered units for tie-breaking.
                score = self._bm25._score_document(
                    list(remaining_units), idx
                )

                if (
                    gain > best_gain
                    or (gain == best_gain and score > best_score)
                    or (gain == best_gain and score == best_score and (best_idx is None or idx < best_idx))
                ):
                    best_gain = gain
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                # No eligible row covers any remaining unit.
                break

            verse_idx = best_idx
            source = self._bm25.source_verses[verse_idx].strip()
            target = self._bm25.target_verses[verse_idx].strip()

            seen_verse_indices.add(verse_idx)
            coverage_examples.append(
                AlignedExample(
                    verse_idx=verse_idx,
                    source=source,
                    target=target,
                    selection_score=best_score,
                    selection_method="coverage",
                    evidence_tier=1,
                )
            )

            # Subtract newly covered units.
            newly_covered = set(
                self._bm25._normalize_text(source).split()
            ) & remaining_units
            remaining_units -= newly_covered

        return tuple(semantic_examples + coverage_examples)
