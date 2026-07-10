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
    When exclude_idx is supplied, every valid corpus row whose normalised
    source-unit SEQUENCE equals the query's normalised source-unit sequence is
    excluded from BOTH stages (equivalence-group source exclusion).  This
    prevents leakage when the same English source verse appears at multiple
    corpus indices (e.g. NUM 7:64 and identical repeated blessing verses).
    The exact exclude_idx is also always excluded regardless of text.

    If exclude_idx is None, no text-based exclusion occurs — all rows are
    eligible (current behaviour preserved).

    Target text is NEVER used for exclusion decisions.

Evidence tier:
    All returned AlignedExample objects carry evidence_tier=1
    (sentence co-occurrence).  This is NOT word alignment; it only means the
    paired sentences are mutual translations.
"""

from __future__ import annotations

from typing import Optional

from constrained_translation.protocol import AlignedExample
from constrained_translation.text_normalize import normalize_source_units
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
        # Precompute immutable per-document unit sets and inverted index.
        # This is the source-only index used by the coverage stage.
        # _doc_units[idx] : frozenset[str]  — normalised tokens for row idx
        # _inverted_index[unit] : list[int] — sorted row indices containing unit
        # ------------------------------------------------------------------
        self._doc_units: dict[int, frozenset[str]] = {
            idx: frozenset(normalize_source_units(self._bm25.source_verses[idx]))
            for idx in self._bm25.valid_indices
        }

        # Precompute normalised source sequence (tuple) per valid index for
        # equivalence-group detection.  Sequence equality (not set equality)
        # ensures we only exclude truly identical source text, not merely
        # overlapping vocabulary.
        self._doc_norm_seq: dict[int, tuple[str, ...]] = {
            idx: tuple(normalize_source_units(self._bm25.source_verses[idx]))
            for idx in self._bm25.valid_indices
        }

        _inv: dict[str, list[int]] = {}
        for idx in self._bm25.valid_indices:
            for unit in self._doc_units[idx]:
                _inv.setdefault(unit, []).append(idx)
        # Sort each postings list so tie-breaking by lowest verse_idx is stable.
        self._inverted_index: dict[str, list[int]] = {
            unit: sorted(postings) for unit, postings in _inv.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        query: str,
        exclude_idx: Optional[int] = None,
        n_semantic: Optional[int] = None,
        n_coverage: Optional[int] = None,
    ) -> tuple[AlignedExample, ...]:
        """Return at most n_semantic + n_coverage deduplicated AlignedExamples.

        Parameters
        ----------
        query:
            Source-language text to find examples for.
        exclude_idx:
            0-based corpus index of the held-out / query verse.  When supplied,
            this verse and ALL corpus rows with an identical normalised source
            sequence are excluded from both stages (equivalence-group source
            exclusion).  Target text is never used for exclusion.
            Pass None to include all verses (no text-based exclusion).
        n_semantic:
            Override the number of semantic (BM25) candidates for this call.
            Defaults to the value passed at construction time.
        n_coverage:
            Override the number of coverage-targeted candidates for this call.
            Defaults to the value passed at construction time.

        Returns
        -------
        tuple[AlignedExample, ...]
            Semantic results first, then non-duplicate coverage results.
            All entries have evidence_tier=1.
        """
        _n_semantic = n_semantic if n_semantic is not None else self._n_semantic
        _n_coverage = n_coverage if n_coverage is not None else self._n_coverage

        # ------------------------------------------------------------------
        # Compute equivalence-group exclusion set (source-sequence based).
        # Only active when exclude_idx is not None.
        # ------------------------------------------------------------------
        equiv_excluded: set[int] = set()
        if exclude_idx is not None:
            # Normalised source sequence of the held-out query verse.
            query_norm_seq = tuple(normalize_source_units(query))
            for idx in self._bm25.valid_indices:
                if self._doc_norm_seq[idx] == query_norm_seq:
                    equiv_excluded.add(idx)
            # Always exclude exclude_idx itself (even if text differs — stable
            # index contract).
            equiv_excluded.add(exclude_idx)

        def _is_excluded(idx: int) -> bool:
            if exclude_idx is None:
                return False
            return idx in equiv_excluded

        # ----------------------------------------------------------------
        # Stage 1 — semantic retrieval via BM25
        # ----------------------------------------------------------------
        # We may need to over-fetch to get enough results after filtering
        # the equivalence group.  Request all valid_indices worth of
        # candidates at most; filter down to _n_semantic after.
        n_total = len(self._bm25.valid_indices)
        fetch_k = min(n_total, _n_semantic + len(equiv_excluded) + 1)

        raw_semantic = self._bm25._simple_search(
            query, top_k=fetch_k, exclude_idx=(exclude_idx if exclude_idx is not None else -1)
        )

        semantic_examples: list[AlignedExample] = []
        seen_verse_indices: set[int] = set()

        for one_based_idx, source, target, score in raw_semantic:
            if len(semantic_examples) >= _n_semantic:
                break
            verse_idx = one_based_idx - 1  # convert to 0-based
            if _is_excluded(verse_idx):
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
        # Compute normalised query units using shared normalizer.
        query_units: set[str] = set(normalize_source_units(query))

        # Subtract units already covered by semantic stage results.
        # Use precomputed _doc_units to avoid re-normalizing corpus docs.
        covered_units: set[str] = set()
        for ex in semantic_examples:
            covered_units |= (
                set(normalize_source_units(ex.source)) & query_units
            )

        remaining_units = query_units - covered_units

        coverage_examples: list[AlignedExample] = []

        while remaining_units and len(coverage_examples) < _n_coverage:
            best_idx: int | None = None
            best_gain: int = 0
            best_score: float = -1.0

            # Use inverted index to find only candidate rows that overlap with
            # remaining_units — avoids scanning the full valid_indices list.
            candidate_indices: set[int] = set()
            for unit in remaining_units:
                if unit in self._inverted_index:
                    candidate_indices.update(self._inverted_index[unit])

            for idx in sorted(candidate_indices):  # sorted for deterministic tie-breaking
                if _is_excluded(idx):
                    continue
                if idx in seen_verse_indices:
                    continue

                # Reuse cached doc units — no normalize call here.
                doc_tokens = self._doc_units[idx]
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

            # Subtract newly covered units using precomputed doc units.
            newly_covered = self._doc_units[verse_idx] & remaining_units
            remaining_units -= newly_covered

        return tuple(semantic_examples + coverage_examples)
