"""tests/constrained_translation/test_experiment_runner.py

Focused offline TDD tests for the experiment runner (paired conditions).
Uses FakeBackend exclusively.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from constrained_translation.experiment.manifest import (
    ManifestItem, build_manifest,
)
from constrained_translation.experiment.runner import (
    ConditionSpec,
    SEMANTIC_ONLY,
    SEMANTIC_COVERAGE,
    run_condition,
    run_language,
    _count_result_lines,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_manifest_files(tmp_path):
    """6-line aligned corpus with 4 good rows → small manifest."""
    eng = [
        "In the beginning God created the heavens and the earth",
        "The earth was without form and void darkness was deep",
        "God said let there be light and there was light",
        "God saw that the light was good and divided it",
        "God called the light Day and the darkness he called Night",
        "And there was evening and there was morning the first day",
    ]
    tgt = [
        "Au commencement Dieu créa les cieux et la terre",
        "La terre était sans forme et vide les ténèbres étaient",
        "Dieu dit que la lumière soit et la lumière fut",
        "Dieu vit que la lumière était bonne et la sépara",
        "Dieu appela la lumière Jour et les ténèbres Nuit",
        "Ce fut le soir et le matin du premier jour",
    ]
    vrefs = [f"GEN 1:{i+1}" for i in range(len(eng))]

    eng_path = tmp_path / "eng.txt"
    tgt_path = tmp_path / "tgt.txt"
    vref_path = tmp_path / "vref.txt"

    eng_path.write_text("\n".join(eng) + "\n", encoding="utf-8")
    tgt_path.write_text("\n".join(tgt) + "\n", encoding="utf-8")
    vref_path.write_text("\n".join(vrefs) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.jsonl"
    items = build_manifest(
        eng_corpus_path=str(eng_path),
        tgt_corpus_path=str(tgt_path),
        vref_path=str(vref_path),
        lang="tst",
        n=3,
        seed=42,
        output_path=manifest_path,
    )

    return {
        "eng_path": str(eng_path),
        "tgt_path": str(tgt_path),
        "vref_path": str(vref_path),
        "manifest_path": manifest_path,
        "manifest_items": items,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# ConditionSpec tests
# ---------------------------------------------------------------------------

class TestConditionSpec:
    def test_semantic_only_spec(self):
        assert SEMANTIC_ONLY.n_semantic == 10
        assert SEMANTIC_ONLY.n_coverage == 0
        assert SEMANTIC_ONLY.max_retries == 0
        assert SEMANTIC_ONLY.name == "semantic_only"

    def test_semantic_coverage_spec(self):
        assert SEMANTIC_COVERAGE.n_semantic == 5
        assert SEMANTIC_COVERAGE.n_coverage == 5
        assert SEMANTIC_COVERAGE.max_retries == 2
        assert SEMANTIC_COVERAGE.name == "semantic_coverage"

    def test_total_budget_equal(self):
        """Semantic-only total (10+0) == Semantic+coverage total (5+5)."""
        so_total = SEMANTIC_ONLY.n_semantic + SEMANTIC_ONLY.n_coverage
        sc_total = SEMANTIC_COVERAGE.n_semantic + SEMANTIC_COVERAGE.n_coverage
        assert so_total == sc_total == 10


# ---------------------------------------------------------------------------
# run_condition tests
# ---------------------------------------------------------------------------

class TestRunCondition:
    def test_writes_results_jsonl(self, tiny_manifest_files, tmp_path):
        out_dir = tmp_path / "cond_out"
        run_condition(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        results_path = out_dir / "results.jsonl"
        assert results_path.exists()
        lines = [l for l in results_path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(tiny_manifest_files["manifest_items"])

    def test_writes_rollup_json(self, tiny_manifest_files, tmp_path):
        out_dir = tmp_path / "cond_out2"
        run_condition(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        rollup_path = out_dir / "rollup.json"
        assert rollup_path.exists()
        data = json.loads(rollup_path.read_text())
        assert "total_units" in data
        assert data["total_units"] == len(tiny_manifest_files["manifest_items"])

    def test_writes_meta_json(self, tiny_manifest_files, tmp_path):
        out_dir = tmp_path / "cond_out3"
        run_condition(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        meta_path = out_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["condition"] == "semantic_only"
        assert meta["lang"] == "tst"
        assert meta["seed"] == 42

    def test_results_have_required_fields(self, tiny_manifest_files, tmp_path):
        out_dir = tmp_path / "cond_out4"
        run_condition(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        for line in (out_dir / "results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for field in ("item_id", "source_text", "translation",
                          "coverage_pass", "retry_count", "hard_failure", "error"):
                assert field in rec, f"Missing field {field!r} in result record"

    def test_resumable_skip(self, tiny_manifest_files, tmp_path):
        """Second call with same dir is skipped (returns skipped=True)."""
        out_dir = tmp_path / "cond_resume"
        kwargs = dict(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        run_condition(**kwargs)
        result = run_condition(**kwargs)
        assert result["skipped"] is True

    def test_force_reruns(self, tiny_manifest_files, tmp_path):
        """force=True re-runs even if complete."""
        out_dir = tmp_path / "cond_force"
        kwargs = dict(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
            force=True,
        )
        run_condition(**kwargs)
        result = run_condition(**kwargs)
        assert result["skipped"] is False

    def test_semantic_coverage_condition(self, tiny_manifest_files, tmp_path):
        """semantic_coverage condition runs without error."""
        out_dir = tmp_path / "cond_sc"
        result = run_condition(
            manifest_items=tiny_manifest_files["manifest_items"],
            condition=SEMANTIC_COVERAGE,
            output_dir=out_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )
        assert result["skipped"] is False
        lines = [l for l in (out_dir / "results.jsonl").read_text().splitlines() if l.strip()]
        assert len(lines) == len(tiny_manifest_files["manifest_items"])


class TestCountResultLines:
    def test_absent_file(self, tmp_path):
        assert _count_result_lines(tmp_path / "no_such_file.jsonl") == 0

    def test_counts_nonempty(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text('{"a":1}\n{"b":2}\n\n{"c":3}\n')
        assert _count_result_lines(f) == 3

    def test_stops_before_truncated_json_record(self, tmp_path):
        """A crash-truncated final write must not make a run look complete."""
        f = tmp_path / "r.jsonl"
        f.write_text('{"a":1}\n{"b":')
        assert _count_result_lines(f) == 1


# ---------------------------------------------------------------------------
# run_language tests
# ---------------------------------------------------------------------------

class TestRunLanguage:
    def test_both_conditions_created(self, tiny_manifest_files, tmp_path):
        lang_dir = tmp_path / "lang_tst"
        run_language(
            lang="tst",
            manifest_path=tiny_manifest_files["manifest_path"],
            lang_output_dir=lang_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            seed=42,
            repo_path=tmp_path,
        )
        assert (lang_dir / "semantic_only" / "results.jsonl").exists()
        assert (lang_dir / "semantic_coverage" / "results.jsonl").exists()

    def test_summary_keys(self, tiny_manifest_files, tmp_path):
        lang_dir = tmp_path / "lang_tst2"
        summary = run_language(
            lang="tst",
            manifest_path=tiny_manifest_files["manifest_path"],
            lang_output_dir=lang_dir,
            eng_corpus_path=tiny_manifest_files["eng_path"],
            tgt_corpus_path=tiny_manifest_files["tgt_path"],
            seed=42,
            repo_path=tmp_path,
        )
        assert "lang" in summary
        assert "semantic_only" in summary
        assert "semantic_coverage" in summary
