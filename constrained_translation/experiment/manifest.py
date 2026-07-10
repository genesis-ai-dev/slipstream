"""constrained_translation.experiment.manifest — Deterministic held-out manifest builder.

Builds a leakage-safe held-out manifest from aligned corpora:

* Reads eng (source) and target language corpora side-by-side.
* Finds common non-empty aligned rows (both eng and target non-empty).
* Applies simple anomaly filters and records excluded counts.
* Deterministically samples *n* rows using *seed* (random.seed).
* Writes JSONL with fields: item_id, source_text, exclude_idx, plus metadata.
* Same 0-based corpus indices across languages for paired comparisons.

Manifest JSONL record schema
-----------------------------
{
  "item_id":       "GEN 1:1",          # vref label
  "source_text":   "In the beginning…", # English source text
  "exclude_idx":   0,                   # 0-based index in the aligned corpus
  "lang":          "mya",              # target language code
  "target_text":   "...",              # target reference (for analysis)
  "corpus_idx":    0                   # original 0-based corpus line index
}

Filter exclusions are captured in a sidecar JSON file.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Anomaly-filter thresholds
# ---------------------------------------------------------------------------

# Minimum character length for a non-trivial verse
_MIN_CHARS = 5
# Maximum ratio of source:target length (by character) to filter outliers
_MAX_LEN_RATIO = 20.0
# Maximum fraction of characters that may be ASCII digits  (verse-number bleed)
_MAX_DIGIT_FRACTION = 0.5


@dataclass
class ManifestItem:
    """One held-out manifest entry."""
    item_id: str          # vref label e.g. "GEN 1:1"
    source_text: str      # English source
    exclude_idx: int      # 0-based index in the aligned corpus (held-out exclusion)
    lang: str             # target language code
    target_text: str      # target reference
    corpus_idx: int       # original 0-based corpus line index (== exclude_idx here)


@dataclass
class FilterStats:
    """Statistics on how many rows were filtered at each stage."""
    total_corpus_rows: int
    empty_source: int
    empty_target: int
    too_short: int
    len_ratio: int
    digit_heavy: int
    eligible: int
    sampled: int


def _load_corpus(path: Path) -> list[str]:
    """Read a corpus file, one verse per line, stripping trailing newlines."""
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def _load_vrefs(path: Path) -> list[str]:
    """Read vref.txt, one ref per line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _digit_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def _passes_filters(src: str, tgt: str) -> tuple[bool, str]:
    """Return (passes, reason_if_not).

    Applies in order: empty, too_short, len_ratio, digit_heavy.
    """
    if not src.strip():
        return False, "empty_source"
    if not tgt.strip():
        return False, "empty_target"
    if len(src.strip()) < _MIN_CHARS or len(tgt.strip()) < _MIN_CHARS:
        return False, "too_short"
    ratio = max(len(src), len(tgt)) / max(min(len(src), len(tgt)), 1)
    if ratio > _MAX_LEN_RATIO:
        return False, "len_ratio"
    if _digit_fraction(src) > _MAX_DIGIT_FRACTION:
        return False, "digit_heavy"
    return True, ""


def build_manifest(
    eng_corpus_path: str | Path,
    tgt_corpus_path: str | Path,
    vref_path: str | Path,
    lang: str,
    n: int,
    seed: int,
    output_path: str | Path,
    stats_path: Optional[str | Path] = None,
    exclude_corpus_indices: Optional[list[int]] = None,
) -> list[ManifestItem]:
    """Build a deterministic held-out manifest for one target language.

    Parameters
    ----------
    eng_corpus_path:
        Path to eng-engULB.txt (source, one verse per line).
    tgt_corpus_path:
        Path to target language corpus (same line-order as eng).
    vref_path:
        Path to benchmarks/data/vref.txt (one vref per line, same order).
    lang:
        Target language code (e.g. "mya").
    n:
        Number of held-out items to sample.
    seed:
        RNG seed for deterministic sampling.
    output_path:
        Where to write the manifest JSONL.
    stats_path:
        Optional path to write filter-stats JSON sidecar.
    exclude_corpus_indices:
        If given, restrict eligible indices to this set (enables same
        held-out rows across languages for paired analysis).

    Returns
    -------
    list[ManifestItem]
        The sampled manifest items.
    """
    eng_lines = _load_corpus(Path(eng_corpus_path))
    tgt_lines = _load_corpus(Path(tgt_corpus_path))
    vrefs = _load_vrefs(Path(vref_path))

    # Align all three to the shortest (should all be 41899 for eBible)
    n_rows = min(len(eng_lines), len(tgt_lines), len(vrefs))

    stats_counts: dict[str, int] = {
        "empty_source": 0,
        "empty_target": 0,
        "too_short": 0,
        "len_ratio": 0,
        "digit_heavy": 0,
    }

    eligible_indices: list[int] = []

    for i in range(n_rows):
        src = eng_lines[i]
        tgt = tgt_lines[i]
        passes, reason = _passes_filters(src, tgt)
        if not passes:
            stats_counts[reason] += 1
        else:
            eligible_indices.append(i)

    # If cross-language indices are supplied, intersect
    if exclude_corpus_indices is not None:
        eligible_set = set(eligible_indices) & set(exclude_corpus_indices)
        eligible_indices = sorted(eligible_set)

    rng = random.Random(seed)
    if n > len(eligible_indices):
        raise ValueError(
            f"Requested n={n} but only {len(eligible_indices)} eligible rows for lang={lang!r}."
        )
    sampled_indices = sorted(rng.sample(eligible_indices, n))

    items: list[ManifestItem] = []
    for idx in sampled_indices:
        item = ManifestItem(
            item_id=vrefs[idx] if idx < len(vrefs) else f"IDX:{idx}",
            source_text=eng_lines[idx],
            exclude_idx=idx,
            lang=lang,
            target_text=tgt_lines[idx],
            corpus_idx=idx,
        )
        items.append(item)

    # Write manifest JSONL
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    # Write filter stats
    filter_stats = FilterStats(
        total_corpus_rows=n_rows,
        empty_source=stats_counts["empty_source"],
        empty_target=stats_counts["empty_target"],
        too_short=stats_counts["too_short"],
        len_ratio=stats_counts["len_ratio"],
        digit_heavy=stats_counts["digit_heavy"],
        eligible=len(eligible_indices),
        sampled=len(items),
    )
    if stats_path is not None:
        sp = Path(stats_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(filter_stats), fh, indent=2)
            fh.write("\n")

    return items


def load_manifest(path: str | Path) -> list[ManifestItem]:
    """Load a manifest JSONL file back into ManifestItem objects."""
    items: list[ManifestItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        items.append(ManifestItem(**d))
    return items
