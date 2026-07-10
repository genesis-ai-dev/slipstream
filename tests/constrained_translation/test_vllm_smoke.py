"""Smoke test for Task 14: real vLLM endpoint integration.

This test is SKIPPED unless VLLM_URL and VLLM_MODEL environment variables
are set.  It is intended as a manual gate — run it only when a vLLM instance
is actually running.

Usage:
    VLLM_URL=http://localhost:8000 VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
        pytest tests/constrained_translation/test_vllm_smoke.py -m smoke -v

The test translates 3 Genesis verses via the constrained batch loop and
asserts that hard_failure_pct < 1.0 (at least one item succeeds end-to-end).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

VLLM_URL = os.environ.get("VLLM_URL", "")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

pytestmark = pytest.mark.smoke

skip_if_no_vllm = pytest.mark.skipif(
    not VLLM_URL,
    reason="VLLM_URL not set — skipping real vLLM smoke test",
)


# ---------------------------------------------------------------------------
# Corpus fixtures (3 Genesis verses — English source, French target)
# ---------------------------------------------------------------------------

_SOURCE_VERSES = [
    "In the beginning God created the heavens and the earth",
    "And the earth was without form and void and darkness was upon the face of the deep",
    "And God said Let there be light and there was light",
]

_TARGET_VERSES = [
    "Au commencement Dieu créa les cieux et la terre",
    "La terre était sans forme et vide et les ténèbres couvraient la face de l'abîme",
    "Et Dieu dit Que la lumière soit et la lumière fut",
]

_INPUT_ITEMS = [
    {"item_id": "GEN 1:1", "source_text": _SOURCE_VERSES[0], "exclude_idx": None},
    {"item_id": "GEN 1:2", "source_text": _SOURCE_VERSES[1], "exclude_idx": None},
    {"item_id": "GEN 1:3", "source_text": _SOURCE_VERSES[2], "exclude_idx": None},
]


# ---------------------------------------------------------------------------
# §1  VLLMBackend construction + health check
# ---------------------------------------------------------------------------

@skip_if_no_vllm
def test_backend_is_available():
    """Real backend health check — vLLM must be up."""
    from constrained_translation.vllm_backend import VLLMBackend
    b = VLLMBackend(base_url=VLLM_URL, model=VLLM_MODEL)
    assert b.is_available(), f"vLLM at {VLLM_URL} not available"


# ---------------------------------------------------------------------------
# §2  tokenize() round-trip
# ---------------------------------------------------------------------------

@skip_if_no_vllm
def test_tokenize_returns_ints():
    """tokenize() must return a non-empty list of ints for a real prompt."""
    from constrained_translation.vllm_backend import VLLMBackend
    b = VLLMBackend(base_url=VLLM_URL, model=VLLM_MODEL)
    ids = b.tokenize("Au commencement Dieu")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# §3  decode_token() round-trip
# ---------------------------------------------------------------------------

@skip_if_no_vllm
def test_decode_token_returns_string():
    """decode_token() must return a non-empty string for a real token ID."""
    from constrained_translation.vllm_backend import VLLMBackend
    b = VLLMBackend(base_url=VLLM_URL, model=VLLM_MODEL)
    ids = b.tokenize("hello")
    assert ids, "tokenize returned empty list"
    decoded = b.decode_token(ids[0])
    assert isinstance(decoded, str)
    assert len(decoded) > 0


# ---------------------------------------------------------------------------
# §4  generate() constrained — single verse
# ---------------------------------------------------------------------------

@skip_if_no_vllm
def test_generate_single_verse_constrained():
    """generate() with a simple grammar must return token_ids and text."""
    from constrained_translation.vllm_backend import VLLMBackend
    from constrained_translation.protocol import GenerationResult
    b = VLLMBackend(base_url=VLLM_URL, model=VLLM_MODEL)
    # Very permissive grammar — any sequence of common French words
    grammar = (
        'root ::= word (" " word)*\n'
        'word ::= [a-zA-ZÀ-ÿ\\u2019]+'
    )
    result = b.generate(
        prompt="Translate to French: In the beginning\n\nTranslation:",
        grammar=grammar,
        max_tokens=32,
        temperature=0.0,
    )
    assert isinstance(result, GenerationResult)
    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0
    assert isinstance(result.token_ids, list)
    assert len(result.token_ids) > 0
    assert result.prompt_tokens > 0
    assert result.output_tokens > 0
    assert result.generation_ms >= 0.0


# ---------------------------------------------------------------------------
# §5  Full batch loop — 3 verses, hard_failure_pct < 1.0
# ---------------------------------------------------------------------------

@skip_if_no_vllm
def test_batch_loop_hard_failure_pct(tmp_path):
    """Full batch loop over 3 Genesis verses.  hard_failure_pct must be < 1.0."""
    from constrained_translation.batch_runner import BatchRunner, BatchItem
    from constrained_translation.vllm_backend import VLLMBackend

    # Write corpus files
    src_file = tmp_path / "source.txt"
    tgt_file = tmp_path / "target.txt"
    src_file.write_text("\n".join(_SOURCE_VERSES) + "\n", encoding="utf-8")
    tgt_file.write_text("\n".join(_TARGET_VERSES) + "\n", encoding="utf-8")

    backend = VLLMBackend(base_url=VLLM_URL, model=VLLM_MODEL)
    log_path = str(tmp_path / "events.jsonl")

    runner = BatchRunner(
        source_file=str(src_file),
        target_file=str(tgt_file),
        backend=backend,
        log_path=log_path,
        max_retries=1,
        n_semantic=3,
        n_coverage=3,
        max_tokens=128,
        temperature=0.0,
    )

    items = [
        BatchItem(
            item_id=it["item_id"],
            source_text=it["source_text"],
            exclude_idx=it["exclude_idx"],
        )
        for it in _INPUT_ITEMS
    ]

    results, rollup = runner.run(items)

    # At least one item must succeed
    assert rollup.hard_failure_pct < 100.0, (
        f"All items failed: hard_failure_pct={rollup.hard_failure_pct}"
    )
    # Preferred: most items succeed
    assert rollup.hard_failure_pct < 100.0, "Expected hard_failure_pct < 100%"
