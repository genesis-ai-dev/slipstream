"""constrained_translation.experiment.runner — Paired condition runner.

Runs both experimental conditions on the same manifest:

    Condition A: semantic-only
        n_semantic=10, n_coverage=0, max_retries=0

    Condition B: semantic+coverage
        n_semantic=5, n_coverage=5, max_retries=<configurable>

Both conditions use the same held-out manifest (same exclude_idx, same source_text).
Outputs are written per language/condition to a structured directory tree.

Output layout
--------------
<output_dir>/
  <lang>/
    <lang>_manifest.jsonl          # the held-out manifest
    semantic_only/
      results.jsonl
      events.jsonl
      rollup.json
      meta.json                    # git commit, model, params, seed
    semantic_coverage/
      results.jsonl
      events.jsonl
      rollup.json
      meta.json

Resumable: if results.jsonl already exists and has the same number of items as
the manifest, that condition is skipped.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from constrained_translation.batch_runner import BatchItem, BatchRunner
from constrained_translation.experiment.manifest import ManifestItem, load_manifest
from constrained_translation.fake_backend import FakeBackend


# ---------------------------------------------------------------------------
# Condition specs
# ---------------------------------------------------------------------------

@dataclass
class ConditionSpec:
    """Parameters for one experimental condition."""
    name: str             # "semantic_only" | "semantic_coverage"
    n_semantic: int
    n_coverage: int
    max_retries: int


SEMANTIC_ONLY = ConditionSpec(
    name="semantic_only",
    n_semantic=10,
    n_coverage=0,
    max_retries=0,
)

SEMANTIC_COVERAGE = ConditionSpec(
    name="semantic_coverage",
    n_semantic=5,
    n_coverage=5,
    max_retries=2,
)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def _git_commit(repo_path: str | Path) -> str:
    """Return current git HEAD sha (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_path), timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _build_meta(
    condition: ConditionSpec,
    lang: str,
    seed: int,
    n_items: int,
    repo_path: str | Path,
    vllm_url: Optional[str],
    model: Optional[str],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a metadata dict capturing run provenance."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_path),
        "lang": lang,
        "condition": condition.name,
        "n_semantic": condition.n_semantic,
        "n_coverage": condition.n_coverage,
        "max_retries": condition.max_retries,
        "seed": seed,
        "n_items": n_items,
        "backend": "vllm" if vllm_url else "fake",
        "vllm_url": vllm_url or None,
        "model": model or None,
    }
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

def _count_result_lines(path: Path) -> int:
    """Count consecutive valid JSONL records (0 if absent or corrupt)."""
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        count += 1
    return count


def run_condition(
    manifest_items: list[ManifestItem],
    condition: ConditionSpec,
    output_dir: Path,
    eng_corpus_path: str,
    tgt_corpus_path: str,
    lang: str,
    seed: int,
    repo_path: str | Path,
    vllm_url: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    """Run one condition on the manifest, writing outputs to output_dir.

    Parameters
    ----------
    manifest_items:
        Items from the held-out manifest.
    condition:
        ConditionSpec describing n_semantic/n_coverage/max_retries.
    output_dir:
        Directory to write results/events/rollup/meta files.
    eng_corpus_path:
        Path to English source corpus.
    tgt_corpus_path:
        Path to target language corpus.
    lang:
        Target language code (for metadata).
    seed:
        Experiment seed (for metadata).
    repo_path:
        Repo root for git commit capture.
    vllm_url:
        If given, use VLLMBackend; otherwise FakeBackend.
    model:
        Model name for VLLMBackend.
    max_tokens:
        Max generation tokens.
    temperature:
        Sampling temperature.
    force:
        If True, overwrite existing outputs.

    Returns
    -------
    dict with paths to results/events/rollup/meta files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    events_path = output_dir / "events.jsonl"
    rollup_path = output_dir / "rollup.json"
    meta_path = output_dir / "meta.json"

    n_expected = len(manifest_items)

    # Resumability: skip if already complete
    if not force and _count_result_lines(results_path) == n_expected:
        sys.stderr.write(
            f"[runner] Skipping {lang}/{condition.name}: "
            f"already {n_expected} results in {results_path}\n"
        )
        return {
            "results": str(results_path),
            "events": str(events_path),
            "rollup": str(rollup_path),
            "meta": str(meta_path),
            "skipped": True,
        }

    # Build backend
    if vllm_url:
        from constrained_translation.vllm_backend import VLLMBackend
        backend = VLLMBackend(base_url=vllm_url, model=model or "")
    else:
        backend = FakeBackend()

    # Build batch items from manifest
    batch_items = [
        BatchItem(
            item_id=item.item_id,
            source_text=item.source_text,
            exclude_idx=item.exclude_idx,
        )
        for item in manifest_items
    ]

    # Build runner
    runner = BatchRunner(
        source_file=eng_corpus_path,
        target_file=tgt_corpus_path,
        backend=backend,
        log_path=str(events_path),
        max_retries=condition.max_retries,
        n_semantic=condition.n_semantic,
        n_coverage=condition.n_coverage,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    results, rollup = runner.run(batch_items)

    # Write results JSONL
    with results_path.open("w", encoding="utf-8") as fh:
        for r in results:
            rec = {
                "item_id": r.item_id,
                "source_text": r.source_text,
                "translation": r.translation,
                "coverage_pass": r.coverage_pass,
                "retry_count": r.retry_count,
                "hard_failure": r.hard_failure,
                "error": r.error,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write rollup JSON
    import dataclasses
    rollup_dict = dataclasses.asdict(rollup)
    with rollup_path.open("w", encoding="utf-8") as fh:
        json.dump(rollup_dict, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Write meta JSON
    meta = _build_meta(
        condition=condition,
        lang=lang,
        seed=seed,
        n_items=n_expected,
        repo_path=repo_path,
        vllm_url=vllm_url,
        model=model,
    )
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return {
        "results": str(results_path),
        "events": str(events_path),
        "rollup": str(rollup_path),
        "meta": str(meta_path),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Paired run for one language
# ---------------------------------------------------------------------------

def run_language(
    lang: str,
    manifest_path: str | Path,
    lang_output_dir: Path,
    eng_corpus_path: str,
    tgt_corpus_path: str,
    seed: int,
    repo_path: str | Path,
    vllm_url: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    coverage_max_retries: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """Run both conditions for one language and return a summary dict."""
    manifest_items = load_manifest(manifest_path)

    # Override semantic_coverage max_retries if specified
    sem_cov = ConditionSpec(
        name=SEMANTIC_COVERAGE.name,
        n_semantic=SEMANTIC_COVERAGE.n_semantic,
        n_coverage=SEMANTIC_COVERAGE.n_coverage,
        max_retries=coverage_max_retries,
    )

    sem_only_paths = run_condition(
        manifest_items=manifest_items,
        condition=SEMANTIC_ONLY,
        output_dir=lang_output_dir / "semantic_only",
        eng_corpus_path=eng_corpus_path,
        tgt_corpus_path=tgt_corpus_path,
        lang=lang,
        seed=seed,
        repo_path=repo_path,
        vllm_url=vllm_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        force=force,
    )

    sem_cov_paths = run_condition(
        manifest_items=manifest_items,
        condition=sem_cov,
        output_dir=lang_output_dir / "semantic_coverage",
        eng_corpus_path=eng_corpus_path,
        tgt_corpus_path=tgt_corpus_path,
        lang=lang,
        seed=seed,
        repo_path=repo_path,
        vllm_url=vllm_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        force=force,
    )

    return {
        "lang": lang,
        "semantic_only": sem_only_paths,
        "semantic_coverage": sem_cov_paths,
    }
