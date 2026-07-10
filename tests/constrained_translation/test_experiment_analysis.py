"""tests/constrained_translation/test_experiment_analysis.py

Focused offline TDD tests for the experiment analysis module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import numpy as np

from constrained_translation.experiment.analysis import (
    _bootstrap_mean_diff_ci,
    _sign_flip_p_value,
    _score_item,
    analyze_language,
    compute_macro,
    render_markdown,
    LangAnalysis,
    MacroAnalysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_results(item_ids: list[str], translations: list[str], tmp_path: Path) -> Path:
    """Write a fake results.jsonl and return the parent dir."""
    path = tmp_path / "results.jsonl"
    with path.open("w") as fh:
        for iid, trans in zip(item_ids, translations):
            rec = {
                "item_id": iid,
                "source_text": "source",
                "translation": trans,
                "coverage_pass": True,
                "retry_count": 0,
                "hard_failure": False,
                "error": None,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return tmp_path


def _make_fake_rollup(tmp_path: Path, **kwargs) -> Path:
    defaults = {
        "total_units": 3,
        "mean_tok_per_sec": 100.0,
        "coverage_pass_pct": 100.0,
        "retry_pct": 0.0,
        "hard_failure_pct": 0.0,
        "token_audit_pass_pct": 100.0,
        "total_input_tokens": 300,
        "total_output_tokens": 30,
    }
    defaults.update(kwargs)
    path = tmp_path / "rollup.json"
    path.write_text(json.dumps(defaults, indent=2))
    return tmp_path


def _make_manifest_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


@pytest.fixture
def simple_analysis_dirs(tmp_path):
    """Three items with known translations and references."""
    item_ids = ["GEN 1:1", "GEN 1:2", "GEN 1:3"]
    references = [
        "Au commencement Dieu créa les cieux et la terre",
        "La terre était sans forme et vide",
        "Dieu dit que la lumière soit",
    ]
    # Semantic-only translations: identical to references (perfect score)
    so_translations = list(references)
    # Semantic+cov translations: slightly different
    sc_translations = [
        "Au commencement Dieu créa les cieux et la terre et les eaux",
        "La terre était sans forme et vide et ténèbres",
        "Dieu dit que la lumière soit et la lumière fut",
    ]

    manifest_items = [
        {
            "item_id": iid, "source_text": f"eng {i}",
            "exclude_idx": i, "lang": "tst",
            "target_text": ref, "corpus_idx": i,
        }
        for i, (iid, ref) in enumerate(zip(item_ids, references))
    ]

    manifest_path = tmp_path / "manifest.jsonl"
    _make_manifest_jsonl(manifest_items, manifest_path)

    so_dir = tmp_path / "semantic_only"
    sc_dir = tmp_path / "semantic_coverage"
    so_dir.mkdir()
    sc_dir.mkdir()

    _make_fake_results(item_ids, so_translations, so_dir)
    _make_fake_results(item_ids, sc_translations, sc_dir)
    _make_fake_rollup(so_dir, coverage_pass_pct=100.0, hard_failure_pct=0.0)
    _make_fake_rollup(sc_dir, coverage_pass_pct=100.0, hard_failure_pct=5.0)

    return {
        "manifest_path": manifest_path,
        "so_dir": so_dir,
        "sc_dir": sc_dir,
        "references": references,
        "so_translations": so_translations,
        "sc_translations": sc_translations,
        "item_ids": item_ids,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_positive_diffs(self):
        """All positive diffs → CI should be entirely positive."""
        diffs = [0.1] * 50
        lo, hi = _bootstrap_mean_diff_ci(diffs, n_boot=500, seed=42)
        assert lo > 0
        assert hi > 0

    def test_all_zero_diffs(self):
        diffs = [0.0] * 20
        lo, hi = _bootstrap_mean_diff_ci(diffs, n_boot=500, seed=42)
        assert abs(lo) < 1e-9
        assert abs(hi) < 1e-9

    def test_ci_ordered(self):
        diffs = [float(i - 5) * 0.1 for i in range(20)]
        lo, hi = _bootstrap_mean_diff_ci(diffs, n_boot=500, seed=42)
        assert lo <= hi

    def test_deterministic(self):
        diffs = [0.05 * i for i in range(10)]
        r1 = _bootstrap_mean_diff_ci(diffs, n_boot=500, seed=42)
        r2 = _bootstrap_mean_diff_ci(diffs, n_boot=500, seed=42)
        assert r1 == r2

    def test_different_seed_may_differ(self):
        """Different seeds generally produce different CIs (not guaranteed but expected)."""
        diffs = [float(i) * 0.01 for i in range(50)]
        r1 = _bootstrap_mean_diff_ci(diffs, n_boot=200, seed=1)
        r2 = _bootstrap_mean_diff_ci(diffs, n_boot=200, seed=999)
        # Not a hard requirement but we expect them to differ with random data
        # (they could coincidentally be equal — we just check the function runs)
        assert isinstance(r1, tuple) and len(r1) == 2
        assert isinstance(r2, tuple) and len(r2) == 2


# ---------------------------------------------------------------------------
# Sign-flip p-value tests
# ---------------------------------------------------------------------------

class TestSignFlipPValue:
    def test_all_positive_small_p(self):
        """Consistent positive diffs → p << 0.05."""
        diffs = [0.1] * 30
        p = _sign_flip_p_value(diffs, n_iter=2000, seed=42)
        assert p < 0.05

    def test_zero_diffs_large_p(self):
        diffs = [0.0] * 20
        p = _sign_flip_p_value(diffs, n_iter=2000, seed=42)
        # Zero diffs → observed stat = 0; all permutations trivially ≥ 0
        # p should be 1.0 (all counts extreme)
        assert p >= 0.99

    def test_deterministic(self):
        diffs = [0.05, -0.02, 0.08, 0.01, -0.03]
        p1 = _sign_flip_p_value(diffs, n_iter=1000, seed=42)
        p2 = _sign_flip_p_value(diffs, n_iter=1000, seed=42)
        assert p1 == p2

    def test_p_in_range(self):
        diffs = [float(i - 5) * 0.05 for i in range(20)]
        p = _sign_flip_p_value(diffs, n_iter=1000, seed=42)
        assert 0.0 < p <= 1.0


# ---------------------------------------------------------------------------
# chrF+ scorer tests
# ---------------------------------------------------------------------------

class TestScoreItem:
    def test_identical(self):
        s = _score_item("Hello world", "Hello world")
        assert abs(s - 1.0) < 1e-6

    def test_empty_hypothesis(self):
        assert _score_item("", "Hello world") == 0.0

    def test_empty_reference(self):
        assert _score_item("Hello world", "") == 0.0

    def test_coverage_failure_artifact(self):
        assert _score_item("text [COVERAGE_FAILURE]", "Hello world") == 0.0

    def test_audit_failure_artifact(self):
        assert _score_item("[AUDIT_FAILURE]", "Hello world") == 0.0

    def test_partial_overlap_positive(self):
        score = _score_item("Hello world", "Hello earth")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# analyze_language tests
# ---------------------------------------------------------------------------

class TestAnalyzeLanguage:
    def test_basic_analysis(self, simple_analysis_dirs):
        d = simple_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        assert result.lang == "tst"
        assert result.n_items == 3
        assert 0.0 <= result.sem_only_mean_chrf <= 1.0
        assert 0.0 <= result.sem_cov_mean_chrf <= 1.0

    def test_perfect_so_has_high_chrf(self, simple_analysis_dirs):
        """Semantic-only translations == references → should score near 1.0."""
        d = simple_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        assert result.sem_only_mean_chrf > 0.9, \
            f"Expected near-perfect chrF+ for identical translations, got {result.sem_only_mean_chrf}"

    def test_items_have_expected_fields(self, simple_analysis_dirs):
        d = simple_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        for item in result.items:
            for field in ("item_id", "sem_only_score", "sem_cov_score",
                          "diff", "reference"):
                assert field in item

    def test_ci_and_p_present(self, simple_analysis_dirs):
        d = simple_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        assert result.diff_ci_lo <= result.diff_ci_hi
        assert 0.0 < result.sign_flip_p <= 1.0

    def test_system_metrics_populated(self, simple_analysis_dirs):
        d = simple_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        assert result.sem_only_coverage_pass_pct == 100.0
        assert result.sem_cov_hard_failure_pct == 5.0
        assert result.sem_only_token_audit_pass_pct == 100.0
        assert result.sem_only_mean_tok_per_sec == 100.0

    def test_hard_failure_diagnostic_artifact_scores_zero(self, simple_analysis_dirs):
        """A loud failure must not receive chrF credit for copied source/UNK text."""
        d = simple_analysis_dirs
        so_results = d["so_dir"] / "results.jsonl"
        rows = [json.loads(line) for line in so_results.read_text(encoding="utf-8").splitlines()]
        rows[0]["hard_failure"] = True
        rows[0]["translation"] = rows[0]["source_text"] + " [UNK:missing]"
        so_results.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=100,
            n_sign_flip=100,
            seed=42,
        )
        assert result.items[0]["sem_only_score"] == 0.0


# ---------------------------------------------------------------------------
# compute_macro tests
# ---------------------------------------------------------------------------

class TestComputeMacro:
    def _make_lang_analysis(self, lang, n=5, so_chrf=0.5, sc_chrf=0.6):
        diffs = [sc_chrf - so_chrf] * n
        items = [
            {"item_id": f"{lang}:{i}", "sem_only_score": so_chrf,
             "sem_cov_score": sc_chrf, "diff": sc_chrf - so_chrf,
             "reference": "ref", "sem_only_translation": "so", "sem_cov_translation": "sc"}
            for i in range(n)
        ]
        return LangAnalysis(
            lang=lang, n_items=n,
            sem_only_mean_chrf=so_chrf, sem_cov_mean_chrf=sc_chrf,
            mean_diff=sc_chrf - so_chrf,
            diff_ci_lo=-0.01, diff_ci_hi=0.15,
            sign_flip_p=0.04,
            sem_only_coverage_pass_pct=100.0, sem_cov_coverage_pass_pct=100.0,
            sem_only_hard_failure_pct=0.0, sem_cov_hard_failure_pct=0.0,
            sem_only_retry_pct=0.0, sem_cov_retry_pct=10.0,
            items=items,
        )

    def test_macro_mean_correct(self):
        analyses = [
            self._make_lang_analysis("mya", so_chrf=0.4, sc_chrf=0.5),
            self._make_lang_analysis("npi", so_chrf=0.6, sc_chrf=0.7),
        ]
        macro = compute_macro(analyses, n_boot=100, n_sign_flip=100, seed=42)
        assert abs(macro.macro_sem_only_chrf - 0.5) < 1e-9
        assert abs(macro.macro_sem_cov_chrf - 0.6) < 1e-9
        assert abs(macro.macro_mean_diff - 0.1) < 1e-9

    def test_macro_languages_list(self):
        analyses = [
            self._make_lang_analysis("mya"),
            self._make_lang_analysis("npi"),
            self._make_lang_analysis("ckb"),
        ]
        macro = compute_macro(analyses, n_boot=100, n_sign_flip=100, seed=42)
        assert set(macro.languages) == {"mya", "npi", "ckb"}

    def test_heterogeneity_sd(self):
        """Heterogeneous langs → nonzero SD."""
        analyses = [
            self._make_lang_analysis("mya", so_chrf=0.3, sc_chrf=0.5),  # diff=0.2
            self._make_lang_analysis("npi", so_chrf=0.5, sc_chrf=0.5),  # diff=0.0
        ]
        macro = compute_macro(analyses, n_boot=100, n_sign_flip=100, seed=42)
        assert macro.lang_diff_sd > 0.0

    def test_pooled_ci_and_p(self):
        analyses = [
            self._make_lang_analysis("mya"),
            self._make_lang_analysis("npi"),
        ]
        macro = compute_macro(analyses, n_boot=200, n_sign_flip=200, seed=42)
        assert macro.pooled_diff_ci_lo <= macro.pooled_diff_ci_hi
        assert 0.0 < macro.pooled_sign_flip_p <= 1.0


# ---------------------------------------------------------------------------
# render_markdown tests
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def _make_mock(self, langs=("mya", "npi")):
        analyses = []
        for lang in langs:
            analyses.append(LangAnalysis(
                lang=lang, n_items=10,
                sem_only_mean_chrf=0.45, sem_cov_mean_chrf=0.50,
                mean_diff=0.05, diff_ci_lo=-0.01, diff_ci_hi=0.11,
                sign_flip_p=0.03,
                sem_only_coverage_pass_pct=95.0, sem_cov_coverage_pass_pct=100.0,
                sem_only_hard_failure_pct=5.0, sem_cov_hard_failure_pct=0.0,
                sem_only_retry_pct=0.0, sem_cov_retry_pct=20.0,
                items=[],
            ))
        macro = MacroAnalysis(
            n_languages=len(langs), languages=list(langs),
            macro_sem_only_chrf=0.45, macro_sem_cov_chrf=0.50,
            macro_mean_diff=0.05, lang_diff_sd=0.02,
            per_lang_mean_diffs={l: 0.05 for l in langs},
            per_lang_sem_only_chrf={l: 0.45 for l in langs},
            per_lang_sem_cov_chrf={l: 0.50 for l in langs},
            pooled_diff_ci_lo=-0.01, pooled_diff_ci_hi=0.11,
            pooled_sign_flip_p=0.03,
        )
        return analyses, macro

    def test_contains_language_sections(self):
        analyses, macro = self._make_mock()
        md = render_markdown(analyses, macro)
        assert "mya" in md
        assert "npi" in md

    def test_contains_macro_section(self):
        analyses, macro = self._make_mock()
        md = render_markdown(analyses, macro)
        assert "Macro" in md

    def test_contains_chrf_values(self):
        analyses, macro = self._make_mock()
        md = render_markdown(analyses, macro)
        assert "0.4500" in md or "0.45" in md

    def test_contains_ci(self):
        analyses, macro = self._make_mock()
        md = render_markdown(analyses, macro)
        assert "CI" in md or "ci" in md.lower()

    def test_heterogeneity_noted(self):
        analyses, macro = self._make_mock()
        md = render_markdown(analyses, macro)
        assert "heterogen" in md.lower() or "SD" in md or "sd" in md.lower()
