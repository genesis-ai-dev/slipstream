"""constrained_translation.experiment.sequencing_manifest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Leakage-safe sequencing manifest builder for the Slipstream Sequencing experiment.

Behaviour
─────────
* Reads the English (source) corpus and one corpus per target language.
* Computes a shared eligible-row set: rows that pass structural filters for
  **every** language (source AND all targets).  This guarantees the same
  corpus indices are used across mya, npi, ckb, and tpi.
* Applies normalised-source equivalence grouping: rows whose normalised source
  text is identical are kept together (placed in the same pool), preventing
  evaluation leakage through paraphrase equivalents.
* Partitions eligible equivalence groups into **four disjoint pools** using a
  single deterministic RNG:
    - seed        (default 10)  — initial translated pool; round 0 context
    - acquisition (default 40)  — sentences selected one-per-round
    - fixed_eval  (default 60)  — held-out evaluation set; never enters any
                                  other pool or the few-shot context
    - remaining                 — all other eligible rows (the "project")
* Computes and stores a stable SHA-256 digest over pool membership.
* Serialises/deserialises as JSON for reproducibility.

Structural marker filtering
───────────────────────────
Uses the same ``_passes_filters`` function from
``constrained_translation.experiment.manifest`` to stay consistent with the
existing held-out manifest builder.

Schema
──────
SequencingPoolItem:
    corpus_idx   int          0-based corpus line index
    item_id      str          vref label e.g. "GEN 1:1"
    source_text  str          English source text
    target_texts dict[str,str] language → target text

SequencingManifest:
    languages    list[str]    ordered list of language codes
    random_seed  int          RNG seed used
    seed_size    int          requested seed pool size (groups, not rows)
    acq_size     int          requested acquisition pool size (groups)
    eval_size    int          requested fixed-eval pool size (groups)
    seed         list[SequencingPoolItem]
    acquisition  list[SequencingPoolItem]
    fixed_eval   list[SequencingPoolItem]
    remaining    list[SequencingPoolItem]
    digest       str          hex SHA-256 of pool indices
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from constrained_translation.experiment.manifest import _passes_filters
from constrained_translation.text_normalize import normalize_source_units


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SequencingPoolItem:
    """One row in a sequencing pool — shared across all languages."""
    corpus_idx: int
    item_id: str
    source_text: str
    target_texts: dict  # lang -> target text


@dataclass
class SequencingManifest:
    """All four disjoint pools plus metadata for one sequencing experiment."""
    languages: list
    random_seed: int
    seed_size: int
    acq_size: int
    eval_size: int
    seed: list  # list[SequencingPoolItem]
    acquisition: list  # list[SequencingPoolItem]
    fixed_eval: list  # list[SequencingPoolItem]
    remaining: list  # list[SequencingPoolItem]
    digest: str = ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def manifest_digest(m: SequencingManifest) -> str:
    """Return a stable hex SHA-256 digest over the sorted corpus indices of each pool.

    The digest covers: for each pool in canonical order (seed, acquisition,
    fixed_eval, remaining), the sorted list of corpus_idx values, plus the
    random_seed and language list.  This uniquely identifies the split.
    """
    pool_data = {
        "random_seed": m.random_seed,
        "languages": sorted(m.languages),
        "seed": sorted(item.corpus_idx for item in m.seed),
        "acquisition": sorted(item.corpus_idx for item in m.acquisition),
        "fixed_eval": sorted(item.corpus_idx for item in m.fixed_eval),
        "remaining": sorted(item.corpus_idx for item in m.remaining),
    }
    canonical = json.dumps(pool_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_sequencing_manifest(m: SequencingManifest, path: str | Path) -> None:
    """Serialise manifest to JSON at *path*; creates parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _item_to_dict(item: SequencingPoolItem) -> dict:
        return {
            "corpus_idx": item.corpus_idx,
            "item_id": item.item_id,
            "source_text": item.source_text,
            "target_texts": item.target_texts,
        }

    data = {
        "languages": m.languages,
        "random_seed": m.random_seed,
        "seed_size": m.seed_size,
        "acq_size": m.acq_size,
        "eval_size": m.eval_size,
        "digest": m.digest,
        "seed": [_item_to_dict(i) for i in m.seed],
        "acquisition": [_item_to_dict(i) for i in m.acquisition],
        "fixed_eval": [_item_to_dict(i) for i in m.fixed_eval],
        "remaining": [_item_to_dict(i) for i in m.remaining],
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sequencing_manifest(path: str | Path) -> SequencingManifest:
    """Deserialise a manifest written by :func:`save_sequencing_manifest`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def _dict_to_item(d: dict) -> SequencingPoolItem:
        return SequencingPoolItem(
            corpus_idx=d["corpus_idx"],
            item_id=d["item_id"],
            source_text=d["source_text"],
            target_texts=d["target_texts"],
        )

    return SequencingManifest(
        languages=data["languages"],
        random_seed=data["random_seed"],
        seed_size=data["seed_size"],
        acq_size=data["acq_size"],
        eval_size=data["eval_size"],
        digest=data.get("digest", ""),
        seed=[_dict_to_item(d) for d in data["seed"]],
        acquisition=[_dict_to_item(d) for d in data["acquisition"]],
        fixed_eval=[_dict_to_item(d) for d in data["fixed_eval"]],
        remaining=[_dict_to_item(d) for d in data["remaining"]],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _norm_key(text: str) -> str:
    """Normalised source key used for equivalence grouping."""
    units = normalize_source_units(text)
    return " ".join(units)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_sequencing_manifest(
    eng_corpus_path: str | Path,
    lang_corpus_paths: dict,  # dict[str, str | Path]
    vref_path: str | Path,
    languages: list,
    seed_size: int = 10,
    acq_size: int = 40,
    eval_size: int = 60,
    random_seed: int = 0,
) -> SequencingManifest:
    """Build a leakage-safe four-pool sequencing manifest.

    Parameters
    ----------
    eng_corpus_path:
        Path to the English (source) corpus, one verse per line.
    lang_corpus_paths:
        Mapping from language code to its corpus path.
    vref_path:
        Path to vref.txt (one verse-reference label per line).
    languages:
        Ordered list of language codes to include.
    seed_size:
        Number of equivalence groups to allocate to the seed pool.
    acq_size:
        Number of equivalence groups to allocate to the acquisition pool.
    eval_size:
        Number of equivalence groups to allocate to the fixed-eval pool.
    random_seed:
        Integer seed for deterministic pool assignment.

    Returns
    -------
    SequencingManifest

    Raises
    ------
    FileNotFoundError
        If any corpus file is missing.
    ValueError
        If corpora have mismatched lengths, or if there are insufficient
        eligible rows for the requested pool sizes.
    """
    eng_path = Path(eng_corpus_path)
    vref_p = Path(vref_path)

    # Load corpora — raises FileNotFoundError if missing
    eng_lines = _load_lines(eng_path)
    vrefs = _load_lines(vref_p)

    lang_lines: dict[str, list[str]] = {}
    for lang in languages:
        p = Path(lang_corpus_paths[lang])
        lang_lines[lang] = _load_lines(p)

    # Alignment check: every language corpus must be at least as long as eng
    n_rows = len(eng_lines)
    for lang in languages:
        if len(lang_lines[lang]) < n_rows:
            raise ValueError(
                f"Language corpus {lang!r} has {len(lang_lines[lang])} rows but "
                f"English corpus has {n_rows} rows — misalignment detected."
            )

    # Build eligible row set: pass _passes_filters for source AND every target
    eligible_indices: list[int] = []
    for i in range(min(n_rows, len(vrefs))):
        src = eng_lines[i]
        # Check source against a dummy target first (corpus marker check)
        # Use a real target to avoid spurious empty-target rejection
        src_ok, _ = _passes_filters(src, "placeholder target text for filter check")
        if not src_ok:
            continue
        # Now check each language's target against the (already-validated) source
        all_tgt_ok = True
        for lang in languages:
            tgt = lang_lines[lang][i]
            tgt_ok, _ = _passes_filters("placeholder source text for filter check", tgt)
            if not tgt_ok:
                all_tgt_ok = False
                break
        if not all_tgt_ok:
            continue
        # Full joint check: also apply the full filter on the real pair
        # (catches empty source, too-short, len-ratio, digit-heavy)
        joint_ok, _ = _passes_filters(src, lang_lines[languages[0]][i])
        if not joint_ok:
            continue
        eligible_indices.append(i)

    # Build normalised-source equivalence groups
    # Each group is a tuple of indices sharing the same normalised source key.
    norm_to_group: dict[str, list[int]] = {}
    for i in eligible_indices:
        key = _norm_key(eng_lines[i])
        norm_to_group.setdefault(key, []).append(i)

    # Flatten to a list of groups (preserving deterministic order by min index)
    groups: list[list[int]] = [
        sorted(idxs) for idxs in norm_to_group.values()
    ]
    # Sort groups by their smallest member for determinism
    groups.sort(key=lambda g: g[0])

    n_groups = len(groups)
    total_needed = seed_size + acq_size + eval_size
    if total_needed > n_groups:
        raise ValueError(
            f"Insufficient eligible equivalence groups: need {total_needed} "
            f"(seed={seed_size} + acq={acq_size} + eval={eval_size}) "
            f"but only {n_groups} groups are eligible after filtering."
        )

    # Deterministic shuffle of group order for pool assignment
    rng = random.Random(random_seed)
    group_order = list(range(n_groups))
    rng.shuffle(group_order)

    seed_group_indices = group_order[:seed_size]
    acq_group_indices = group_order[seed_size: seed_size + acq_size]
    eval_group_indices = group_order[seed_size + acq_size: seed_size + acq_size + eval_size]
    rem_group_indices = group_order[seed_size + acq_size + eval_size:]

    def _groups_to_items(gidxs: list[int]) -> list[SequencingPoolItem]:
        items = []
        for gi in gidxs:
            for corpus_idx in groups[gi]:
                items.append(SequencingPoolItem(
                    corpus_idx=corpus_idx,
                    item_id=vrefs[corpus_idx] if corpus_idx < len(vrefs) else f"IDX:{corpus_idx}",
                    source_text=eng_lines[corpus_idx],
                    target_texts={lang: lang_lines[lang][corpus_idx] for lang in languages},
                ))
        # Sort by corpus_idx for stable output
        items.sort(key=lambda it: it.corpus_idx)
        return items

    seed_items = _groups_to_items(seed_group_indices)
    acq_items = _groups_to_items(acq_group_indices)
    eval_items = _groups_to_items(eval_group_indices)
    rem_items = _groups_to_items(rem_group_indices)

    manifest = SequencingManifest(
        languages=list(languages),
        random_seed=random_seed,
        seed_size=seed_size,
        acq_size=acq_size,
        eval_size=eval_size,
        seed=seed_items,
        acquisition=acq_items,
        fixed_eval=eval_items,
        remaining=rem_items,
        digest="",
    )
    manifest.digest = manifest_digest(manifest)
    return manifest
