"""constrained_translation.experiment.corpus_coverage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deterministic corpus coverage accounting for the Slipstream Sequencing experiment.

Behaviour
─────────
1. Reuses ``normalize_source_units`` exactly — no second tokenizer.
2. Freezes the operational project as the first ``limit`` (default 200) items
   from ``manifest.remaining`` in ascending corpus_idx order.  Raises ValueError
   for invalid limits (≤ 0).
3. Builds an occurrence-frequency index over the fixed project slice.
4. Given current translated source cells, computes:
   - known_types / unknown_types (normalized)
   - weighted_unknown_occurrences (sum of freq_index[t] for t in unknown_types)
   - weighted_supported_occurrences (sum of freq_index[t] for t in known_types)
   - support_percentage (100 × supported / total)
   - fully_supported_count (project cells where ALL normalized types are known)
   - unsupported_spans per project cell (UNKSpan-compatible)
   - unsupported_units per project cell (frozenset of unknown normalized types)
5. detect_false_source_abstentions: finds units marked unknown that ARE
   canonically present in the translated source pool (normalization/index defect).
6. marginal_gain: frequency-weighted gain + unweighted new-type count for a
   candidate against the current known set and project frequency index.

Design
──────
- Pure Python; no GPU, no model, no target text.
- All functions are deterministic given the same inputs.
- Unsupported spans are UNKSpan instances (from constrained_translation.protocol),
  compatible with UNKDetector semantics, source-side only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.protocol import UNKSpan
from constrained_translation.text_normalize import normalize_source_units
from constrained_translation.unk_detector import UNKDetector as _UNKDetector

_unk_detector = _UNKDetector()


# ---------------------------------------------------------------------------
# Default project slice size (per spec: first 200 items from remaining)
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_LIMIT = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectSlice:
    """The frozen operational project view: first *limit* items from remaining,
    ordered by corpus_idx, plus their pre-computed frequency index."""

    items: list[SequencingPoolItem]
    limit: int
    freq_index: dict[str, int]  # normalized_type → total occurrence count in slice


@dataclass
class CoverageStats:
    """Coverage accounting result for one (project_slice, translated_pool) pair."""

    known_types: frozenset[str]
    unknown_types: frozenset[str]
    weighted_unknown_occurrences: int
    weighted_supported_occurrences: int
    support_percentage: float
    fully_supported_count: int
    # Per-item unsupported spans (UNKSpan-compatible), keyed by corpus_idx
    unsupported_spans: dict[int, tuple[UNKSpan, ...]]
    # Per-item frozenset of unknown normalized types, keyed by corpus_idx
    unsupported_units: dict[int, frozenset[str]]


# ---------------------------------------------------------------------------
# build_project_slice
# ---------------------------------------------------------------------------

def build_project_slice(
    manifest: SequencingManifest,
    limit: int = DEFAULT_PROJECT_LIMIT,
) -> ProjectSlice:
    """Return the deterministic operational project slice.

    Sorts ``manifest.remaining`` by ``corpus_idx`` ascending, takes the first
    ``limit`` items (or all if fewer are available), and builds the
    occurrence-frequency index using ``normalize_source_units``.

    Parameters
    ----------
    manifest:
        The sequencing manifest whose ``remaining`` pool defines the project.
    limit:
        Maximum number of items to include.  Must be ≥ 1.

    Returns
    -------
    ProjectSlice
        Frozen project view with items in corpus_idx order and freq_index.

    Raises
    ------
    ValueError
        If ``limit`` ≤ 0.
    """
    if limit <= 0:
        raise ValueError(f"limit must be ≥ 1, got {limit!r}")

    # Sort by corpus_idx, take first `limit`
    sorted_items = sorted(manifest.remaining, key=lambda it: it.corpus_idx)
    items = sorted_items[:limit]

    # Build occurrence-frequency index
    freq_index: dict[str, int] = {}
    for item in items:
        for unit in normalize_source_units(item.source_text):
            freq_index[unit] = freq_index.get(unit, 0) + 1

    return ProjectSlice(items=items, limit=limit, freq_index=freq_index)


# ---------------------------------------------------------------------------
# Internal helper: build known_types from translated source cells
# ---------------------------------------------------------------------------

def _build_known_types(translated_source_cells: Sequence[str]) -> frozenset[str]:
    """Return the frozenset of normalized types seen in the translated pool."""
    known: set[str] = set()
    for cell in translated_source_cells:
        known.update(normalize_source_units(cell))
    return frozenset(known)


# ---------------------------------------------------------------------------
# Internal helper: unsupported spans for one source string given known types
# ---------------------------------------------------------------------------

def _unsupported_spans_for(
    source_text: str,
    known_types: frozenset[str],
) -> tuple[UNKSpan, ...]:
    """Return UNKSpan tuples for contiguous runs of uncovered tokens in source_text.

    Delegates to ``UNKDetector.detect`` with ``known_types`` as the attested
    vocabulary.  Since ``known_types`` contains already-normalized units (output
    of ``normalize_source_units``), the normalization applied inside UNKDetector
    is idempotent, and the results are identical to a direct lookup.  This
    eliminates the previously duplicated tokenization/merge algorithm.

    Source-side only — no target input.
    """
    return _unk_detector.detect(source_text, known_types)


# ---------------------------------------------------------------------------
# compute_coverage_stats
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    project_slice: ProjectSlice,
    translated_source_cells: Sequence[str],
) -> CoverageStats:
    """Compute coverage statistics for a project slice given translated source cells.

    Parameters
    ----------
    project_slice:
        The operational project view (from ``build_project_slice``).
    translated_source_cells:
        Source texts of cells that have been translated so far.  These are
        normalized via ``normalize_source_units`` to build the known-type set.
        No target/reference text is required or accepted.

    Returns
    -------
    CoverageStats
        All coverage accounting fields.

    Notes
    -----
    - ``support_percentage`` is vacuously 100.0 for an empty project
      (no unknowns possible).
    - ``fully_supported_count`` counts project items where every normalized
      source type (from ``normalize_source_units``) is in ``known_types``.
    - Unsupported spans use UNKSpan (protocol-compatible) and are source-side only.
    """
    freq_index = project_slice.freq_index
    known_types = _build_known_types(translated_source_cells)

    all_project_types = frozenset(freq_index.keys())
    unknown_types = all_project_types - known_types

    weighted_unknown = sum(freq_index[t] for t in unknown_types)
    weighted_supported = sum(freq_index[t] for t in known_types if t in freq_index)

    total_weight = weighted_unknown + weighted_supported
    if total_weight == 0:
        support_percentage = 100.0
    else:
        support_percentage = 100.0 * weighted_supported / total_weight

    # Per-item accounting
    fully_supported_count = 0
    unsupported_spans: dict[int, tuple[UNKSpan, ...]] = {}
    unsupported_units: dict[int, frozenset[str]] = {}

    for item in project_slice.items:
        item_types = frozenset(normalize_source_units(item.source_text))
        item_unknown = item_types - known_types
        unsupported_units[item.corpus_idx] = item_unknown
        if not item_unknown:
            fully_supported_count += 1
        # Build unsupported spans for this item
        unsupported_spans[item.corpus_idx] = _unsupported_spans_for(
            item.source_text, known_types
        )

    return CoverageStats(
        known_types=known_types,
        unknown_types=unknown_types,
        weighted_unknown_occurrences=weighted_unknown,
        weighted_supported_occurrences=weighted_supported,
        support_percentage=support_percentage,
        fully_supported_count=fully_supported_count,
        unsupported_spans=unsupported_spans,
        unsupported_units=unsupported_units,
    )


# ---------------------------------------------------------------------------
# detect_false_source_abstentions
# ---------------------------------------------------------------------------

def detect_false_source_abstentions(
    stats: CoverageStats,
    translated_source_cells: Sequence[str],
) -> dict[int, frozenset[str]]:
    """Detect false source abstentions: unknown units that ARE in the translated pool.

    A false abstention occurs when a project item's unit is marked unknown but
    its normalized form IS canonically present in the translated source evidence.
    This indicates a normalization/index defect (per spec §Source abstention artifact).

    Parameters
    ----------
    stats:
        Coverage stats (must include ``unsupported_units``).
    translated_source_cells:
        The translated source pool (same pool used to compute ``stats``).
        Source-side only; no target text.

    Returns
    -------
    dict[int, frozenset[str]]
        Maps corpus_idx → frozenset of falsely-abstained normalized types.
        Only nonempty entries are returned; items with no false abstentions
        do not appear as keys.
    """
    translated_types = _build_known_types(translated_source_cells)
    result: dict[int, frozenset[str]] = {}
    for corpus_idx, unknown_set in stats.unsupported_units.items():
        false_abs = unknown_set & translated_types
        if false_abs:
            result[corpus_idx] = false_abs
    return result


# ---------------------------------------------------------------------------
# marginal_gain
# ---------------------------------------------------------------------------

def marginal_gain(
    candidate_source_text: str,
    project_slice: ProjectSlice,
    known_types: frozenset[str],
) -> tuple[float, int]:
    """Compute the frequency-weighted marginal gain and new-type count for a candidate.

    Parameters
    ----------
    candidate_source_text:
        Raw source text of the candidate sentence.  Normalized via
        ``normalize_source_units`` before lookup.
    project_slice:
        The operational project slice (for ``freq_index``).
    known_types:
        Current set of known normalized types.

    Returns
    -------
    (weighted_gain, new_type_count)
        weighted_gain: sum of ``freq_index[t]`` for each newly covered type ``t``
            that appears in ``freq_index`` (project-present types only).
        new_type_count: count of normalized types in candidate NOT in ``known_types``
            AND present in ``project_slice.freq_index``.  Types absent from the
            project yield zero contribution to both gains.

    Notes
    -----
    - Duplicate types within a candidate sentence are deduplicated (set semantics).
    - Types in candidate but absent from ``freq_index`` contribute 0 to both
      ``weighted_gain`` and ``new_type_count`` (project-gain semantics).
    """
    candidate_types = frozenset(normalize_source_units(candidate_source_text))
    freq_index = project_slice.freq_index
    # Restrict to project-present new types only
    new_project_types = frozenset(
        t for t in candidate_types
        if t not in known_types and t in freq_index
    )

    weighted_gain = sum(freq_index[t] for t in new_project_types)
    new_type_count = len(new_project_types)

    return float(weighted_gain), new_type_count
