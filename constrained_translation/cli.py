"""constrained_translation.cli — Command-line interface for constrained translation.

Usage — offline (FakeBackend)
------------------------------
python -m constrained_translation.cli \\
    --source  corpus/source.txt \\
    --target  corpus/target.txt \\
    --input   items.jsonl \\
    --output  results.jsonl \\
    --log     events.jsonl \\
    --fake-backend

Usage — real vLLM backend
--------------------------
python -m constrained_translation.cli \\
    --source  corpus/source.txt \\
    --target  corpus/target.txt \\
    --input   items.jsonl \\
    --output  results.jsonl \\
    --log     events.jsonl \\
    --vllm-url http://localhost:8000 \\
    --model   Qwen/Qwen2.5-7B-Instruct

Backend selection flags (mutually exclusive):
  --fake-backend          Use FakeBackend (no GPU, offline, for testing)
  --vllm-url URL          Use VLLMBackend pointing at this URL

  --model MODEL           Required when --vllm-url is given; the model name
                          as registered in vLLM.

Input JSONL format (one JSON object per line):

    {"item_id": "GEN 1:1", "source_text": "In the beginning...", "exclude_idx": 0}
    {"item_id": "GEN 1:2", "source_text": "The earth...",        "exclude_idx": null}

All three fields (``item_id``, ``source_text``, ``exclude_idx``) are required.
``exclude_idx`` must be present as a key; use JSON ``null`` for no held-out
exclusion.  Silent inference of exclude_idx is never performed.

Output JSONL format (one JSON object per line per input item):

    {"item_id": ..., "source_text": ..., "translation": ...,
     "coverage_pass": ..., "retry_count": ..., "hard_failure": ..., "error": ...}

Stdout:
    A single JSON object with aggregate batch rollup statistics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from constrained_translation.batch_runner import BatchItem, BatchRunner
from constrained_translation.fake_backend import FakeBackend
from constrained_translation.protocol import BatchRollup, TranslationResult


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert dataclass instances and frozensets to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_serialisable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    return obj


def _result_to_dict(r: TranslationResult) -> dict:
    """Convert a TranslationResult to a JSON-safe dict for the output JSONL."""
    return {
        "item_id": r.item_id,
        "source_text": r.source_text,
        "translation": r.translation,
        "coverage_pass": r.coverage_pass,
        "retry_count": r.retry_count,
        "hard_failure": r.hard_failure,
        "error": r.error,
    }


def _rollup_to_dict(rollup: BatchRollup) -> dict:
    """Convert a BatchRollup dataclass to a plain dict."""
    return dataclasses.asdict(rollup)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input_jsonl(path: str) -> list[BatchItem]:
    """Parse input JSONL into a list of BatchItem records.

    Raises
    ------
    ValueError
        If any record is missing ``item_id``, ``source_text``, or
        ``exclude_idx`` (all three are required; ``exclude_idx`` may be null).
    """
    items: list[BatchItem] = []
    raw = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {lineno}: invalid JSON — {exc}") from exc

        # Validate required fields
        missing = []
        for field in ("item_id", "source_text", "exclude_idx"):
            if field not in record:
                missing.append(field)
        if missing:
            raise ValueError(
                f"Line {lineno}: missing required field(s): {', '.join(missing)}"
            )

        items.append(
            BatchItem(
                item_id=str(record["item_id"]),
                source_text=str(record["source_text"]),
                exclude_idx=record["exclude_idx"],
            )
        )
    return items


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    p = argparse.ArgumentParser(
        prog="constrained_translation.cli",
        description="Constrained few-shot translation batch processor.",
    )

    # Required I/O arguments
    p.add_argument(
        "--source", required=True, metavar="PATH",
        help="Path to the aligned source corpus file (one verse per line).",
    )
    p.add_argument(
        "--target", required=True, metavar="PATH",
        help="Path to the aligned target corpus file (same order as --source).",
    )
    p.add_argument(
        "--input", required=True, metavar="PATH",
        help="Path to input JSONL file (item_id, source_text, exclude_idx required).",
    )
    p.add_argument(
        "--output", required=True, metavar="PATH",
        help="Path to write output results JSONL (one record per input item).",
    )
    p.add_argument(
        "--log", required=True, metavar="PATH",
        help="Path to write event log JSONL (coverage, audit, provenance events).",
    )

    # Backend selection (mutually exclusive: --fake-backend or --vllm-url)
    backend_group = p.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--fake-backend", action="store_true", default=False,
        help="Use FakeBackend for offline testing (no GPU or model download required).",
    )
    backend_group.add_argument(
        "--vllm-url", metavar="URL", default=None,
        help=(
            "Use VLLMBackend pointing at this URL "
            "(e.g. http://localhost:8000).  Requires --model."
        ),
    )

    # --model is required when --vllm-url is given; accepted but ignored for --fake-backend
    p.add_argument(
        "--model", metavar="MODEL", default=None,
        help=(
            "Model name as registered in vLLM "
            "(required when --vllm-url is given; "
            "e.g. Qwen/Qwen2.5-7B-Instruct)."
        ),
    )

    # Runner hyperparameters
    p.add_argument(
        "--max-retries", type=int, default=3, metavar="N",
        help="Maximum coverage-expansion retries per item (default: 3).",
    )
    p.add_argument(
        "--n-semantic", type=int, default=5, metavar="N",
        help="Number of semantic (BM25) examples per item (default: 5).",
    )
    p.add_argument(
        "--n-coverage", type=int, default=5, metavar="N",
        help="Number of coverage-targeted examples per item (default: 5).",
    )
    p.add_argument(
        "--max-tokens", type=int, default=256, metavar="N",
        help="Maximum tokens to generate per item (default: 256).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0, metavar="F",
        help="Sampling temperature (default: 0.0 = greedy).",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI main entry point.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- Select backend -------------------------------------------------------
    if args.fake_backend:
        backend = FakeBackend()
    elif args.vllm_url:
        # Validate that --model was also supplied
        if not args.model:
            sys.stderr.write(
                "Error: --vllm-url requires --model to be specified.\n"
                "Example: --vllm-url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct\n"
            )
            return 1
        from constrained_translation.vllm_backend import VLLMBackend
        backend = VLLMBackend(base_url=args.vllm_url, model=args.model)
    else:
        # Default to FakeBackend when no backend flag is given
        sys.stderr.write(
            "Warning: no backend selected; defaulting to --fake-backend.\n"
            "Pass --fake-backend explicitly, or use --vllm-url URL --model MODEL.\n"
        )
        backend = FakeBackend()

    # -- Parse input ----------------------------------------------------------
    try:
        items = _parse_input_jsonl(args.input)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Error reading input: {exc}\n")
        return 1

    if not items:
        sys.stderr.write("Error: input JSONL is empty — no items to translate.\n")
        return 1

    # -- Build runner ---------------------------------------------------------
    runner = BatchRunner(
        source_file=args.source,
        target_file=args.target,
        backend=backend,
        log_path=args.log,
        max_retries=args.max_retries,
        n_semantic=args.n_semantic,
        n_coverage=args.n_coverage,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # -- Run ------------------------------------------------------------------
    try:
        results, rollup = runner.run(items)
    except RuntimeError as exc:
        sys.stderr.write(f"Runtime error: {exc}\n")
        return 1
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1

    # -- Write output JSONL ---------------------------------------------------
    output_path = Path(args.output)
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")
    except OSError as exc:
        sys.stderr.write(f"Error writing output: {exc}\n")
        return 1

    # -- Print rollup JSON to stdout ------------------------------------------
    rollup_dict = _rollup_to_dict(rollup)
    sys.stdout.write(json.dumps(rollup_dict, ensure_ascii=False) + "\n")
    sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
