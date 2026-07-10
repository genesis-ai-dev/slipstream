"""constrained_translation.experiment.policies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source-only acquisition policies for the Slipstream Sequencing experiment.

Three deterministic, leakage-free policies for selecting the next candidate
to translate each round.  All policies:
  - Accept only ``AcquisitionCandidate`` objects (no target_texts).
  - Are pure functions: no global state mutation, no side effects.
  - Remove exactly one candidate per invocation (caller is responsible for
    removing the chosen candidate from the pool between rounds).
  - Raise ``ValueError`` for an empty candidate list.

Policies
--------
arbitrary_policy(candidates, seed)
    Deterministic permutation via an isolated ``random.Random(seed)``; no
    global RNG contamination.  A fully fresh seeded RNG is created for
    each call; the call position within the permutation is derived from the
    current remaining-pool composition (sorted corpus_idx tuple) so that
    removing the winner and re-calling produces the next item in the
    seeded shuffle.

silver_policy(candidates, translated_source_texts)
    Three-level ordering (best → worst):
    1. Highest fraction of normalized source-unit types in candidate that
       are already known from ``translated_source_texts``.
       Fraction = |known ∩ candidate_types| / |candidate_types|.
       Punctuation-only candidate (empty type set): fraction defined as 0.0.
    2. Highest source-side BM25 similarity to the translated-source pool
       (pool is treated as one concatenated document for IDF; each
       candidate is scored against the pool).
    3. Lowest corpus_idx (stable).

    BM25 parameters: k1=1.5, b=0.75 (same as BM25Query defaults).
    The pool documents are each translated_source_text treated as one
    document; the candidate is the query.  IDF is corpus-based over the
    translated pool documents; TF is over the candidate (query-side).

golden_policy(candidates, project_slice, known_types)
    Three-level ordering (best → worst):
    1. Max frequency-weighted marginal gain against ``project_slice``.
       Reuses ``corpus_coverage.marginal_gain`` exactly.
    2. Max unweighted new project-type count (from marginal_gain).
    3. Lowest corpus_idx (stable).

golden_policy_with_gain(candidates, project_slice, known_types) → GoldenGain
    Same as golden_policy but also returns the ex-ante weighted and
    unweighted gain values for logging/calibration.

AcquisitionCandidate
    Leakage-safe dataclass: corpus_idx, item_id, source_text only.
    No target_texts, no reference.

candidates_from_manifest(manifest) → list[AcquisitionCandidate]
    Adapter: strips target data from manifest.acquisition items.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from constrained_translation.experiment.corpus_coverage import (
    ProjectSlice,
    marginal_gain,
)
from constrained_translation.experiment.sequencing_manifest import SequencingManifest
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# AcquisitionCandidate — leakage-safe dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcquisitionCandidate:
    """Source-only candidate for acquisition.

    Contains only the fields needed by source-side policies.  Target data
    (``target_texts``, translations, references) is deliberately absent to
    prevent target leakage into acquisition decisions.
    """

    corpus_idx: int
    item_id: str
    source_text: str


# ---------------------------------------------------------------------------
# Adapter: SequencingManifest → list[AcquisitionCandidate]
# ---------------------------------------------------------------------------

def candidates_from_manifest(manifest: SequencingManifest) -> list[AcquisitionCandidate]:
    """Return AcquisitionCandidates from manifest.acquisition, stripping target data.

    Parameters
    ----------
    manifest:
        A SequencingManifest produced by build_sequencing_manifest.

    Returns
    -------
    list[AcquisitionCandidate]
        One candidate per acquisition item, in the same order as
        ``manifest.acquisition``.  No target data is carried over.
    """
    return [
        AcquisitionCandidate(
            corpus_idx=item.corpus_idx,
            item_id=item.item_id,
            source_text=item.source_text,
        )
        for item in manifest.acquisition
    ]


# ---------------------------------------------------------------------------
# Arbitrary policy
# ---------------------------------------------------------------------------

def arbitrary_policy(
    candidates: list[AcquisitionCandidate],
    *,
    seed: int,
) -> AcquisitionCandidate:
    """Return the next candidate in a deterministic seeded permutation.

    Each call creates a fresh isolated ``random.Random(seed)`` instance —
    global RNG state is never read or written.  The position within the
    permutation is determined by shuffling the candidates using the seed;
    the first item of the shuffled list that is still in ``candidates``
    (by sorted-corpus_idx position) is returned.

    Specifically: sort ``candidates`` by corpus_idx, shuffle a copy with
    ``random.Random(seed).shuffle``, then return the first element.  This
    guarantees that across rounds (as candidates are removed one by one),
    the policy replays the same seeded order each time given the
    ever-shrinking pool.

    Parameters
    ----------
    candidates:
        Current remaining acquisition candidates (non-empty).
    seed:
        Integer seed.  Supported values: any int (at least 0–9999 tested).

    Returns
    -------
    AcquisitionCandidate
        The next candidate to translate.

    Raises
    ------
    ValueError
        If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("arbitrary_policy: candidates list is empty")

    # Sort by corpus_idx for a stable canonical ordering before shuffling.
    sorted_pool = sorted(candidates, key=lambda c: c.corpus_idx)

    # Isolated RNG — never touches global random state.
    rng = random.Random(seed)
    rng.shuffle(sorted_pool)

    return sorted_pool[0]


# ---------------------------------------------------------------------------
# Silver policy — internal BM25 helper
# ---------------------------------------------------------------------------

def _bm25_score_candidate_vs_pool(
    candidate_text: str,
    pool_texts: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Source-side BM25 score: candidate as query, pool as corpus.

    Parameters
    ----------
    candidate_text:
        The candidate source text (query).
    pool_texts:
        Translated source texts forming the corpus.
    k1, b:
        BM25 parameters (matching BM25Query defaults).

    Returns
    -------
    float
        Aggregated BM25 score.  Returns 0.0 for an empty pool.

    Notes
    -----
    Normalization uses ``normalize_source_units`` (NFKC + lower + strip
    punctuation + split), which is compatible with BM25Query._normalize_text
    on ASCII text and adds correct Unicode/RTL handling.
    """
    if not pool_texts:
        return 0.0

    # Tokenize pool documents
    docs: list[list[str]] = [normalize_source_units(t) for t in pool_texts]
    n_docs = len(docs)

    # Document lengths
    doc_lengths = [len(d) for d in docs]
    avg_doc_length = sum(doc_lengths) / n_docs if n_docs else 0.0

    # Document-frequency: how many docs contain each term
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            doc_freq[term] += 1

    # Term-frequency per document
    term_freqs: list[Counter[str]] = [Counter(doc) for doc in docs]

    # Query: normalize candidate text
    query_terms = normalize_source_units(candidate_text)
    if not query_terms:
        return 0.0

    score = 0.0
    for doc_idx, (doc_len, tf) in enumerate(zip(doc_lengths, term_freqs)):
        for term in set(query_terms):
            if term not in tf:
                continue
            tf_val = tf[term]
            df = doc_freq[term]
            # BM25 IDF (with +1 smoothing as in BM25Query)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf_val * (k1 + 1)
            denominator = tf_val + k1 * (1 - b + b * (doc_len / avg_doc_length))
            score += idf * (numerator / denominator)

    return score


def _known_fraction(candidate_types: frozenset[str], known_types: frozenset[str]) -> float:
    """Fraction of candidate's normalized types that are in known_types.

    Returns 0.0 for an empty candidate type set (punctuation-only).
    """
    if not candidate_types:
        return 0.0
    return len(candidate_types & known_types) / len(candidate_types)


# ---------------------------------------------------------------------------
# Silver policy
# ---------------------------------------------------------------------------

def silver_policy(
    candidates: list[AcquisitionCandidate],
    *,
    translated_source_texts: Sequence[str],
) -> AcquisitionCandidate:
    """Choose the next candidate by source-side coverage ordering.

    Three-level priority (higher = better):
    1. ``known_fraction``: fraction of normalized candidate types present in
       the translated source pool.  Punctuation-only candidates score 0.0.
    2. BM25 similarity (k1=1.5, b=0.75) of candidate to translated pool.
       The pool documents serve as the BM25 corpus; the candidate is the
       query.  Empty pool → all scores 0.0 → falls through to tiebreak.
    3. Lowest ``corpus_idx`` (deterministic stable tiebreak).

    Parameters
    ----------
    candidates:
        Current remaining acquisition candidates (non-empty).
    translated_source_texts:
        Source texts of already-translated cells.  Source-side only — no
        target/reference text is accepted or used.

    Returns
    -------
    AcquisitionCandidate

    Raises
    ------
    ValueError
        If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("silver_policy: candidates list is empty")

    # Build known types from translated source texts
    known_types: frozenset[str] = frozenset(
        unit
        for text in translated_source_texts
        for unit in normalize_source_units(text)
    )

    def _score(c: AcquisitionCandidate) -> tuple[float, float, int]:
        c_types = frozenset(normalize_source_units(c.source_text))
        frac = _known_fraction(c_types, known_types)
        bm25 = _bm25_score_candidate_vs_pool(c.source_text, translated_source_texts)
        # Higher frac and bm25 are better; lower corpus_idx is better (negate for max)
        return (frac, bm25, -c.corpus_idx)

    return max(candidates, key=_score)


# ---------------------------------------------------------------------------
# GoldenGain — result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldenGain:
    """Ex-ante gain information for the chosen golden candidate.

    Attributes
    ----------
    candidate:
        The chosen AcquisitionCandidate.
    weighted_gain:
        Frequency-weighted marginal gain (sum of project freq_index values
        for newly covered types).
    unweighted_gain:
        Unweighted new-type count (number of project types newly covered).
    """

    candidate: AcquisitionCandidate
    weighted_gain: float
    unweighted_gain: int


# ---------------------------------------------------------------------------
# Golden policy
# ---------------------------------------------------------------------------

def _golden_key(
    c: AcquisitionCandidate,
    project_slice: ProjectSlice,
    known_types: frozenset[str],
) -> tuple[float, int, int]:
    """Compute sorting key for golden policy (higher = better)."""
    wg, ug = marginal_gain(c.source_text, project_slice, known_types)
    return (wg, ug, -c.corpus_idx)


def golden_policy(
    candidates: list[AcquisitionCandidate],
    *,
    project_slice: ProjectSlice,
    known_types: frozenset[str],
) -> AcquisitionCandidate:
    """Choose the next candidate by frequency-weighted project gain.

    Three-level priority (higher = better):
    1. Max frequency-weighted marginal gain against ``project_slice``.
       Uses ``corpus_coverage.marginal_gain`` exactly.
       Types absent from the project yield zero gain.
    2. Max unweighted new-type count (number of project types newly covered).
    3. Lowest ``corpus_idx`` (deterministic stable tiebreak).

    Parameters
    ----------
    candidates:
        Current remaining acquisition candidates (non-empty).
    project_slice:
        The frozen 200-line (or configured-size) operational project slice.
        Source-side only — no target data.
    known_types:
        Current set of known normalized source types.

    Returns
    -------
    AcquisitionCandidate

    Raises
    ------
    ValueError
        If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("golden_policy: candidates list is empty")

    return max(candidates, key=lambda c: _golden_key(c, project_slice, known_types))


def golden_policy_with_gain(
    candidates: list[AcquisitionCandidate],
    *,
    project_slice: ProjectSlice,
    known_types: frozenset[str],
) -> GoldenGain:
    """Same as ``golden_policy`` but also returns ex-ante gain for logging.

    Parameters
    ----------
    candidates:
        Current remaining acquisition candidates (non-empty).
    project_slice:
        The frozen operational project slice.
    known_types:
        Current set of known normalized source types.

    Returns
    -------
    GoldenGain
        Contains the chosen candidate plus its weighted and unweighted gain.

    Raises
    ------
    ValueError
        If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("golden_policy_with_gain: candidates list is empty")

    chosen = golden_policy(candidates, project_slice=project_slice, known_types=known_types)
    wg, ug = marginal_gain(chosen.source_text, project_slice, known_types)
    return GoldenGain(candidate=chosen, weighted_gain=wg, unweighted_gain=ug)
