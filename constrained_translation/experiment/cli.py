"""constrained_translation.experiment.cli — Experiment harness CLI.

Subcommands
-----------
prepare
    Build held-out manifests for all languages.

    python -m constrained_translation.experiment.cli prepare \\
        --corpus-dir   Corpus/ \\
        --vref         benchmarks/data/vref.txt \\
        --output-dir   experiment_out/ \\
        --languages    mya npi ckb som tpi ton \\
        --n            10 \\
        --seed         42

run
    Run both conditions (semantic-only and semantic+coverage) for all
    languages using the prepared manifests.

    # Offline (FakeBackend):
    python -m constrained_translation.experiment.cli run \\
        --output-dir   experiment_out/ \\
        --corpus-dir   Corpus/ \\
        --languages    mya npi ckb som tpi ton \\
        --seed         42

    # Live vLLM:
    python -m constrained_translation.experiment.cli run \\
        --output-dir   experiment_out/ \\
        --corpus-dir   Corpus/ \\
        --languages    mya npi ckb som tpi ton \\
        --seed         42 \\
        --vllm-url     http://localhost:8000 \\
        --model        Qwen/Qwen2.5-7B-Instruct \\
        --coverage-max-retries 2

analyze
    Analyze paired results and produce JSON + Markdown reports.

    python -m constrained_translation.experiment.cli analyze \\
        --output-dir   experiment_out/ \\
        --languages    mya npi ckb som tpi ton \\
        --seed         42

Notes
-----
* ``--n`` specifies the number of held-out items per language.
  Pilot: n=10 or n=20 before full n=60.
* Semantic-only condition: n_semantic=10, n_coverage=0, max_retries=0.
* Semantic+coverage condition: n_semantic=5, n_coverage=5,
  max_retries=<--coverage-max-retries, default 2>.
* Run is resumable: completed conditions (complete results.jsonl) are skipped.
* No fine-tuning, no unconstrained fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_LANGUAGES = ["mya", "npi", "ckb", "som", "tpi", "ton"]

_ENG_CORPUS_FILENAME = "eng-engULB.txt"

_TGT_CORPUS_FILENAMES = {
    "mya": "mya-mya.txt",
    "npi": "npi-npiulb.txt",
    "ckb": "ckb-ckb.txt",
    "som": "som-som.txt",
    "tpi": "tpi-tpi.txt",
    "ton": "ton-ton.txt",
}


def _tgt_corpus_path(corpus_dir: Path, lang: str) -> str:
    filename = _TGT_CORPUS_FILENAMES.get(lang, f"{lang}-{lang}.txt")
    return str(corpus_dir / filename)


# ---------------------------------------------------------------------------
# prepare subcommand
# ---------------------------------------------------------------------------

def _cmd_prepare(args: argparse.Namespace) -> int:
    from constrained_translation.experiment.manifest import build_manifest

    corpus_dir = Path(args.corpus_dir)
    vref_path = Path(args.vref)
    output_dir = Path(args.output_dir)
    languages = args.languages
    n = args.n
    seed = args.seed

    eng_path = corpus_dir / _ENG_CORPUS_FILENAME
    if not eng_path.exists():
        sys.stderr.write(f"Error: English corpus not found: {eng_path}\n")
        return 1
    if not vref_path.exists():
        sys.stderr.write(f"Error: vref.txt not found: {vref_path}\n")
        return 1

    # First pass: find eligible indices for each language, then intersect
    # for cross-language pairing.
    from constrained_translation.experiment.manifest import (
        _load_corpus, _passes_filters, _load_vrefs,
    )

    eng_lines = _load_corpus(eng_path)
    vrefs = _load_vrefs(vref_path)
    n_rows = min(len(eng_lines), len(vrefs))

    # Find indices eligible across ALL requested languages
    import random as _random

    shared_eligible: Optional[list[int]] = None
    for lang in languages:
        tgt_path_str = _tgt_corpus_path(corpus_dir, lang)
        tgt_path = Path(tgt_path_str)
        if not tgt_path.exists():
            sys.stderr.write(f"Error: target corpus not found: {tgt_path}\n")
            return 1
        tgt_lines = _load_corpus(tgt_path)
        lang_eligible = []
        for i in range(min(n_rows, len(tgt_lines))):
            passes, _ = _passes_filters(eng_lines[i], tgt_lines[i])
            if passes:
                lang_eligible.append(i)
        if shared_eligible is None:
            shared_eligible = lang_eligible
        else:
            shared_eligible = sorted(set(shared_eligible) & set(lang_eligible))

    if shared_eligible is None or len(shared_eligible) < n:
        sys.stderr.write(
            f"Error: insufficient shared eligible rows ({len(shared_eligible or [])} < n={n}).\n"
        )
        return 1

    sys.stderr.write(
        f"[prepare] {len(shared_eligible)} shared eligible rows across {len(languages)} languages.\n"
    )

    # Now build manifest for each language using the shared eligible pool
    for lang in languages:
        lang_dir = output_dir / lang
        manifest_path = lang_dir / f"{lang}_manifest.jsonl"
        stats_path = lang_dir / f"{lang}_filter_stats.json"
        tgt_path_str = _tgt_corpus_path(corpus_dir, lang)

        items = build_manifest(
            eng_corpus_path=str(eng_path),
            tgt_corpus_path=tgt_path_str,
            vref_path=str(vref_path),
            lang=lang,
            n=n,
            seed=seed,
            output_path=manifest_path,
            stats_path=stats_path,
            exclude_corpus_indices=shared_eligible,
        )
        sys.stderr.write(
            f"[prepare] {lang}: wrote {len(items)} items → {manifest_path}\n"
        )

    sys.stderr.write("[prepare] Done.\n")
    return 0


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    from constrained_translation.experiment.runner import run_language

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    languages = args.languages
    seed = args.seed
    vllm_url = getattr(args, "vllm_url", None)
    model = getattr(args, "model", None)
    coverage_max_retries = args.coverage_max_retries
    max_tokens = args.max_tokens
    temperature = args.temperature
    force = args.force

    eng_path = str(corpus_dir / _ENG_CORPUS_FILENAME)

    for lang in languages:
        manifest_path = output_dir / lang / f"{lang}_manifest.jsonl"
        if not manifest_path.exists():
            sys.stderr.write(
                f"Error: manifest not found for {lang}: {manifest_path}\n"
                f"Run 'prepare' first.\n"
            )
            return 1

        tgt_path = _tgt_corpus_path(corpus_dir, lang)
        lang_dir = output_dir / lang

        sys.stderr.write(f"[run] Running language: {lang}\n")
        summary = run_language(
            lang=lang,
            manifest_path=manifest_path,
            lang_output_dir=lang_dir,
            eng_corpus_path=eng_path,
            tgt_corpus_path=tgt_path,
            seed=seed,
            repo_path=Path(__file__).parent.parent.parent,
            vllm_url=vllm_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            coverage_max_retries=coverage_max_retries,
            force=force,
        )
        sem_only_skip = summary["semantic_only"].get("skipped", False)
        sem_cov_skip = summary["semantic_coverage"].get("skipped", False)
        sys.stderr.write(
            f"[run] {lang}: semantic_only={'skipped' if sem_only_skip else 'done'}, "
            f"semantic_coverage={'skipped' if sem_cov_skip else 'done'}\n"
        )

    sys.stderr.write("[run] Done.\n")
    return 0


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> int:
    from constrained_translation.experiment.analysis import run_analysis

    output_dir = Path(args.output_dir)
    languages = args.languages
    seed = args.seed
    n_boot = args.n_boot
    n_sign_flip = args.n_sign_flip

    sys.stderr.write(f"[analyze] Analyzing {len(languages)} languages…\n")
    lang_analyses, macro = run_analysis(
        output_dir=output_dir,
        languages=languages,
        n_boot=n_boot,
        n_sign_flip=n_sign_flip,
        seed=seed,
    )

    analysis_dir = output_dir / "analysis"
    sys.stderr.write(f"[analyze] Wrote analysis to {analysis_dir}\n")

    # Print a brief summary to stdout
    summary = {
        "macro_sem_only_chrf": macro.macro_sem_only_chrf,
        "macro_sem_cov_chrf": macro.macro_sem_cov_chrf,
        "macro_mean_diff": macro.macro_mean_diff,
        "pooled_diff_ci": [macro.pooled_diff_ci_lo, macro.pooled_diff_ci_hi],
        "pooled_sign_flip_p": macro.pooled_sign_flip_p,
        "lang_diff_sd": macro.lang_diff_sd,
        "per_lang": {
            a.lang: {
                "sem_only_chrf": a.sem_only_mean_chrf,
                "sem_cov_chrf": a.sem_cov_mean_chrf,
                "diff": a.mean_diff,
                "p": a.sign_flip_p,
            }
            for a in lang_analyses
        },
    }
    sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    sys.stderr.write("[analyze] Done.\n")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level experiment CLI parser."""
    p = argparse.ArgumentParser(
        prog="constrained_translation.experiment.cli",
        description=(
            "Reproducible multilingual experiment harness for paired "
            "semantic-only vs semantic+coverage evaluation."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── prepare ──────────────────────────────────────────────────────────
    prep = sub.add_parser(
        "prepare",
        help=(
            "Build deterministic held-out manifests from aligned corpora. "
            "Finds shared non-empty eligible rows across all requested languages, "
            "applies anomaly filters, and samples n items per language."
        ),
    )
    prep.add_argument(
        "--corpus-dir", required=True, metavar="DIR",
        help="Directory containing eng-engULB.txt and target language corpora.",
    )
    prep.add_argument(
        "--vref", required=True, metavar="PATH",
        help="Path to benchmarks/data/vref.txt (one vref label per line).",
    )
    prep.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root directory for experiment outputs (created if absent).",
    )
    prep.add_argument(
        "--languages", nargs="+", default=_DEFAULT_LANGUAGES, metavar="LANG",
        help=f"Target language codes (default: {' '.join(_DEFAULT_LANGUAGES)}).",
    )
    prep.add_argument(
        "--n", type=int, default=10, metavar="N",
        help="Number of held-out items per language (default: 10; pilot: 10 or 20).",
    )
    prep.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="RNG seed for deterministic sampling (default: 42).",
    )

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help=(
            "Run both conditions (semantic-only: n_sem=10,n_cov=0,retries=0; "
            "semantic+coverage: n_sem=5,n_cov=5,retries=<--coverage-max-retries>) "
            "on the prepared manifests. Resumable: skips complete outputs."
        ),
    )
    run_p.add_argument(
        "--corpus-dir", required=True, metavar="DIR",
        help="Directory containing corpora (same as prepare).",
    )
    run_p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root experiment output directory (same as prepare).",
    )
    run_p.add_argument(
        "--languages", nargs="+", default=_DEFAULT_LANGUAGES, metavar="LANG",
        help=f"Languages to run (default: {' '.join(_DEFAULT_LANGUAGES)}).",
    )
    run_p.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Seed for metadata (default: 42).",
    )
    run_p.add_argument(
        "--vllm-url", metavar="URL", default=None,
        dest="vllm_url",
        help="vLLM server URL (e.g. http://localhost:8000). Omit for FakeBackend.",
    )
    run_p.add_argument(
        "--model", metavar="MODEL", default=None,
        help="Model name (required when --vllm-url is given).",
    )
    run_p.add_argument(
        "--coverage-max-retries", type=int, default=2, metavar="N",
        help="Max retries for semantic+coverage condition (default: 2).",
    )
    run_p.add_argument(
        "--max-tokens", type=int, default=256, metavar="N",
        help="Max generation tokens (default: 256).",
    )
    run_p.add_argument(
        "--temperature", type=float, default=0.0, metavar="F",
        help="Sampling temperature (default: 0.0).",
    )
    run_p.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing complete outputs (default: skip).",
    )

    # ── analyze ──────────────────────────────────────────────────────────
    ana = sub.add_parser(
        "analyze",
        help=(
            "Analyze paired outputs against target references using chrF+. "
            "Produces per-language and macro JSON/Markdown with bootstrap CIs "
            "and sign-flip p-values."
        ),
    )
    ana.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root experiment output directory (same as prepare/run).",
    )
    ana.add_argument(
        "--languages", nargs="+", default=_DEFAULT_LANGUAGES, metavar="LANG",
        help=f"Languages to analyze (default: {' '.join(_DEFAULT_LANGUAGES)}).",
    )
    ana.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="RNG seed for bootstrap/sign-flip (default: 42).",
    )
    ana.add_argument(
        "--n-boot", type=int, default=2000, metavar="N",
        help="Bootstrap resamples for CI (default: 2000).",
    )
    ana.add_argument(
        "--n-sign-flip", type=int, default=10000, metavar="N",
        help="Sign-flip randomization iterations for p-value (default: 10000).",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        return _cmd_prepare(args)
    elif args.command == "run":
        return _cmd_run(args)
    elif args.command == "analyze":
        return _cmd_analyze(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
