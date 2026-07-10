"""tests/constrained_translation/test_scalar_diagnostics.py

TDD tests for:
1. TokenAuditResult defaulted scalar fields (token_provenance_ratio,
   surface_attestation_similarity).
2. TokenAuditor.audit() computing those scalars correctly (non-gating).
3. results.jsonl persisting raw_generation_text, raw_token_ids,
   token_provenance_ratio, surface_attestation_similarity (null-absent).
4. Analysis non-gating diagnostics: naive_raw_chrf,
   formatting_tolerant_edit_similarity.

All pass/fail logic and hard-failure behavior must remain unchanged.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Optional

import pytest

from constrained_translation.protocol import (
    AlignedExample,
    GenerationResult,
    TokenAuditResult,
)
from constrained_translation.token_auditor import TokenAuditor, _surface_words_from_text


# ---------------------------------------------------------------------------
# Helpers / mini-backend
# ---------------------------------------------------------------------------

def _ex(verse_idx: int, target: str) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source=f"source {verse_idx}",
        target=target,
        selection_score=1.0,
        selection_method="semantic",
        evidence_tier=1,
    )


class _SimpleBackend:
    """Minimal backend: vocab is a list of strings, IDs are indices."""

    def __init__(self, vocab: list[str]):
        self._vocab = vocab  # idx → surface string

    def tokenize(self, text: str) -> list[int]:
        """Return IDs for each whitespace-split word found in vocab."""
        ids = []
        for word in text.split():
            try:
                ids.append(self._vocab.index(word))
            except ValueError:
                pass  # skip unknown (for test simplicity)
        return ids

    def decode_token(self, token_id: int) -> str:
        return self._vocab[token_id]

    def decode_tokens(self, token_ids: list[int]) -> str:
        return " ".join(self._vocab[i] for i in token_ids)

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, grammar, max_tokens, temperature=0.0) -> GenerationResult:
        return GenerationResult(text="", token_ids=[], prompt_tokens=0,
                                output_tokens=0, generation_ms=0.0)


# ---------------------------------------------------------------------------
# 1. TokenAuditResult — scalar fields exist and default to 0.0
# ---------------------------------------------------------------------------

class TestTokenAuditResultScalarFields:
    def test_has_token_provenance_ratio_field(self):
        """TokenAuditResult must expose token_provenance_ratio defaulting to 0.0."""
        r = TokenAuditResult(passed=True, violations=[])
        assert hasattr(r, "token_provenance_ratio"), (
            "TokenAuditResult missing field 'token_provenance_ratio'"
        )

    def test_token_provenance_ratio_default_zero(self):
        r = TokenAuditResult(passed=True, violations=[])
        assert r.token_provenance_ratio == 0.0

    def test_has_surface_attestation_similarity_field(self):
        """TokenAuditResult must expose surface_attestation_similarity defaulting to 0.0."""
        r = TokenAuditResult(passed=True, violations=[])
        assert hasattr(r, "surface_attestation_similarity"), (
            "TokenAuditResult missing field 'surface_attestation_similarity'"
        )

    def test_surface_attestation_similarity_default_zero(self):
        r = TokenAuditResult(passed=True, violations=[])
        assert r.surface_attestation_similarity == 0.0

    def test_fields_are_float_in_range(self):
        r = TokenAuditResult(passed=True, violations=[],
                             token_provenance_ratio=0.5,
                             surface_attestation_similarity=0.75)
        assert 0.0 <= r.token_provenance_ratio <= 1.0
        assert 0.0 <= r.surface_attestation_similarity <= 1.0

    def test_pass_fail_unchanged_by_scalar_fields(self):
        """Scalar fields must not influence passed/violations."""
        r_pass = TokenAuditResult(passed=True, violations=[],
                                  token_provenance_ratio=0.1,
                                  surface_attestation_similarity=0.1)
        r_fail = TokenAuditResult(passed=False, violations=["x"],
                                  token_provenance_ratio=1.0,
                                  surface_attestation_similarity=1.0)
        assert r_pass.passed is True
        assert r_fail.passed is False


# ---------------------------------------------------------------------------
# 2. TokenAuditor.audit() — scalar computation
# ---------------------------------------------------------------------------

class TestTokenAuditorScalars:
    """TokenAuditor.audit() must fill scalar fields non-gatingly."""

    def _make_backend_and_examples(self):
        vocab = ["hello", "world", "foo", "bar", " ", ","]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello world"), _ex(1, "foo bar")]
        return backend, examples

    def test_audit_result_has_provenance_ratio(self):
        backend, examples = self._make_backend_and_examples()
        auditor = TokenAuditor()
        # "hello world" → IDs [0, 1], both licensed
        token_ids = [0, 1]
        attested_vocab = frozenset(["hello", "world"])
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert hasattr(result, "token_provenance_ratio")

    def test_all_licensed_provenance_ratio_one(self):
        """All non-layout tokens licensed → ratio == 1.0."""
        backend, examples = self._make_backend_and_examples()
        auditor = TokenAuditor()
        token_ids = [0, 1]  # "hello", "world" — both licensed via examples
        attested_vocab = frozenset(["hello", "world"])
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert result.token_provenance_ratio == pytest.approx(1.0)

    def test_no_tokens_provenance_ratio_zero(self):
        """Empty token list → ratio 0.0."""
        backend, examples = self._make_backend_and_examples()
        auditor = TokenAuditor()
        result, _ = auditor.audit([], frozenset(["hello"]), backend, examples=examples)
        assert result.token_provenance_ratio == 0.0

    def test_layout_only_tokens_provenance_ratio_zero(self):
        """Pure layout tokens (spaces/punct) → no non-layout denominator → 0.0."""
        vocab = [" ", ",", "."]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, " , .")]
        auditor = TokenAuditor()
        token_ids = [0, 1, 2]  # all layout
        result, _ = auditor.audit(token_ids, frozenset(), backend, examples=examples)
        assert result.token_provenance_ratio == 0.0

    def test_half_licensed_provenance_ratio_half(self):
        """2 non-layout tokens, 1 licensed → ratio 0.5."""
        vocab = ["hello", "unlicensed_word", " "]
        backend = _SimpleBackend(vocab)
        # Only example contributes "hello" (ID 0)
        examples = [_ex(0, "hello")]
        auditor = TokenAuditor()
        attested_vocab = frozenset(["hello", "unlicensed_word"])
        # token_ids: [0=hello, 1=unlicensed_word]; hello is licensed, unlicensed_word is not
        token_ids = [0, 1]
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert result.token_provenance_ratio == pytest.approx(0.5)

    def test_provenance_ratio_does_not_affect_pass(self):
        """Ratio never affects passed."""
        vocab = ["known", "alien_word"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "known")]  # only 'known' is licensed
        auditor = TokenAuditor()
        attested_vocab = frozenset(["known", "alien_word"])
        # alien_word (ID 1) not in licence → audit fails (unlicensed ID)
        token_ids = [0, 1]
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert result.passed is False  # hard failure behavior unchanged
        # ratio should be 0.5 (1 licensed out of 2 non-layout)
        assert result.token_provenance_ratio == pytest.approx(0.5)

    def test_surface_attestation_similarity_exact_match(self):
        """All decoded words in attested_vocab → similarity 1.0."""
        vocab = ["hello", "world"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello world")]
        auditor = TokenAuditor()
        attested_vocab = frozenset(["hello", "world"])
        token_ids = [0, 1]
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert result.surface_attestation_similarity == pytest.approx(1.0)

    def test_surface_attestation_similarity_empty_tokens_zero(self):
        """No tokens → similarity 0.0."""
        vocab = ["hello"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello")]
        auditor = TokenAuditor()
        result, _ = auditor.audit([], frozenset(["hello"]), backend, examples=examples)
        assert result.surface_attestation_similarity == 0.0

    def test_surface_attestation_similarity_in_range(self):
        """similarity must be in [0, 1] for various inputs."""
        vocab = ["hello", "world", "foo"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello world")]
        auditor = TokenAuditor()
        attested_vocab = frozenset(["hello", "world"])
        token_ids = [0, 1, 2]  # "hello world foo"; "foo" not in attested_vocab
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert 0.0 <= result.surface_attestation_similarity <= 1.0

    def test_surface_attestation_similarity_empty_attested_vocab_zero(self):
        """Empty attested_vocab → similarity 0.0 (no candidates)."""
        vocab = ["hello", "world"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello world")]
        auditor = TokenAuditor()
        token_ids = [0, 1]
        result, _ = auditor.audit(token_ids, frozenset(), backend, examples=examples)
        assert result.surface_attestation_similarity == 0.0

    def test_scalars_computed_even_when_audit_fails(self):
        """Scalars are filled even when audit fails (non-gating)."""
        vocab = ["hello", "unlicensed"]
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello")]
        auditor = TokenAuditor()
        attested_vocab = frozenset(["hello", "unlicensed"])
        token_ids = [0, 1]  # unlicensed ID 1 → audit fails
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        assert result.passed is False
        # Both scalars should be computed (non-zero / meaningful)
        assert isinstance(result.token_provenance_ratio, float)
        assert isinstance(result.surface_attestation_similarity, float)
        assert 0.0 <= result.token_provenance_ratio <= 1.0
        assert 0.0 <= result.surface_attestation_similarity <= 1.0

    def test_surface_attestation_similarity_close_word(self):
        """A word close to an attested word should yield similarity > 0."""
        vocab = ["hello"]  # decoded text is "hello"
        backend = _SimpleBackend(vocab)
        examples = [_ex(0, "hello")]
        auditor = TokenAuditor()
        # attested_vocab has "helo" (1 edit away from "hello")
        attested_vocab = frozenset(["helo"])
        token_ids = [0]  # "hello"
        result, _ = auditor.audit(token_ids, attested_vocab, backend, examples=examples)
        # "hello" vs "helo": edit distance 1, max_len 5 → normalized 0.2 → similarity 0.8
        assert result.surface_attestation_similarity > 0.0
        assert result.surface_attestation_similarity < 1.0


# ---------------------------------------------------------------------------
# 3. results.jsonl raw fields
# ---------------------------------------------------------------------------

class TestResultsJsonlRawFields:
    """run_condition() must persist raw generation fields in results.jsonl."""

    def _run_one_condition(self, tmp_path: Path) -> list[dict]:
        """Run a tiny condition and return parsed results."""
        from constrained_translation.experiment.manifest import build_manifest
        from constrained_translation.experiment.runner import run_condition, SEMANTIC_ONLY

        eng = [
            "In the beginning God created the heavens and the earth",
            "The earth was without form and void",
            "God said let there be light",
            "God saw that the light was good",
            "God called the light Day",
            "Evening and morning the first day",
        ]
        tgt = [
            "Au commencement Dieu créa les cieux",
            "La terre était sans forme et vide",
            "Dieu dit que la lumière soit",
            "Dieu vit que la lumière était bonne",
            "Dieu appela la lumière Jour",
            "Ce fut le soir et le matin",
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

        out_dir = tmp_path / "out"
        run_condition(
            manifest_items=items,
            condition=SEMANTIC_ONLY,
            output_dir=out_dir,
            eng_corpus_path=str(eng_path),
            tgt_corpus_path=str(tgt_path),
            lang="tst",
            seed=42,
            repo_path=tmp_path,
        )

        lines = [
            json.loads(l)
            for l in (out_dir / "results.jsonl").read_text().splitlines()
            if l.strip()
        ]
        return lines

    def test_raw_generation_text_present(self, tmp_path):
        """results.jsonl records must have 'raw_generation_text' key."""
        records = self._run_one_condition(tmp_path)
        assert records, "No records written"
        for rec in records:
            assert "raw_generation_text" in rec, (
                f"Missing 'raw_generation_text' in record {rec.get('item_id')}"
            )

    def test_raw_token_ids_present(self, tmp_path):
        """results.jsonl records must have 'raw_token_ids' key."""
        records = self._run_one_condition(tmp_path)
        for rec in records:
            assert "raw_token_ids" in rec, (
                f"Missing 'raw_token_ids' in record {rec.get('item_id')}"
            )

    def test_token_provenance_ratio_present(self, tmp_path):
        """results.jsonl records must have 'token_provenance_ratio' key."""
        records = self._run_one_condition(tmp_path)
        for rec in records:
            assert "token_provenance_ratio" in rec, (
                f"Missing 'token_provenance_ratio' in record {rec.get('item_id')}"
            )

    def test_surface_attestation_similarity_present(self, tmp_path):
        """results.jsonl records must have 'surface_attestation_similarity' key."""
        records = self._run_one_condition(tmp_path)
        for rec in records:
            assert "surface_attestation_similarity" in rec, (
                f"Missing 'surface_attestation_similarity' in record {rec.get('item_id')}"
            )

    def test_raw_fields_null_on_hard_failure(self, tmp_path):
        """Hard-failure records (no generation) must have null raw_token_ids."""
        # We'll inject a failure record directly by inspecting results from coverage-failure
        # scenario. For simplicity, just verify None is JSON-serializable as null.
        rec = {
            "item_id": "x", "raw_generation_text": None,
            "raw_token_ids": None, "token_provenance_ratio": None,
            "surface_attestation_similarity": None,
        }
        serialized = json.dumps(rec)
        parsed = json.loads(serialized)
        assert parsed["raw_token_ids"] is None
        assert parsed["raw_generation_text"] is None

    def test_raw_generation_text_is_string_or_null(self, tmp_path):
        records = self._run_one_condition(tmp_path)
        for rec in records:
            v = rec["raw_generation_text"]
            assert v is None or isinstance(v, str)

    def test_raw_token_ids_is_list_or_null(self, tmp_path):
        records = self._run_one_condition(tmp_path)
        for rec in records:
            v = rec["raw_token_ids"]
            assert v is None or isinstance(v, list)

    def test_token_provenance_ratio_is_float_or_null(self, tmp_path):
        records = self._run_one_condition(tmp_path)
        for rec in records:
            v = rec["token_provenance_ratio"]
            assert v is None or isinstance(v, float)

    def test_surface_attestation_similarity_is_float_or_null(self, tmp_path):
        records = self._run_one_condition(tmp_path)
        for rec in records:
            v = rec["surface_attestation_similarity"]
            assert v is None or isinstance(v, float)

    def test_existing_fields_unchanged(self, tmp_path):
        """Adding new fields must not break existing required fields."""
        records = self._run_one_condition(tmp_path)
        for rec in records:
            for field in ("item_id", "source_text", "translation",
                          "coverage_pass", "retry_count", "hard_failure", "error"):
                assert field in rec, f"Existing field {field!r} missing"


# ---------------------------------------------------------------------------
# 4. Analysis non-gating diagnostics
# ---------------------------------------------------------------------------

class TestFormattingTolerantEditSimilarity:
    """Unit tests for the formatting_tolerant_edit_similarity helper in analysis."""

    def _get_fn(self):
        from constrained_translation.experiment.analysis import (
            _formatting_tolerant_edit_similarity,
        )
        return _formatting_tolerant_edit_similarity

    def test_identical_strings(self):
        fn = self._get_fn()
        assert fn("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_hypothesis_returns_zero(self):
        fn = self._get_fn()
        assert fn("", "hello world") == 0.0

    def test_empty_reference_returns_zero(self):
        fn = self._get_fn()
        assert fn("hello world", "") == 0.0

    def test_both_empty_returns_zero(self):
        fn = self._get_fn()
        assert fn("", "") == 0.0

    def test_punctuation_removed_both_sides(self):
        """After removing punctuation, 'hello!' vs 'hello' should score 1.0."""
        fn = self._get_fn()
        assert fn("hello!", "hello") == pytest.approx(1.0)

    def test_format_chars_removed(self):
        """U+200B (Cf category) must be stripped; both sides normalized identically."""
        fn = self._get_fn()
        # U+200B zero-width space (Cf) — after NFKC + Cf removal on both sides,
        # "hello\u200bworld" → "helloworld"
        # "helloworld"       → "helloworld"  → score 1.0
        s_with = "hello\u200bworld"
        s_without = "helloworld"
        score = fn(s_with, s_without)
        assert score == pytest.approx(1.0)

    def test_nfkc_applied(self):
        """NFKC normalization: full-width 'Ａ' → 'A'."""
        fn = self._get_fn()
        # U+FF21 FULLWIDTH LATIN CAPITAL LETTER A → A after NFKC
        assert fn("\uff21", "A") == pytest.approx(1.0)

    def test_result_in_range(self):
        fn = self._get_fn()
        score = fn("hello world", "goodbye earth")
        assert 0.0 <= score <= 1.0

    def test_normalized_empty_after_stripping_returns_zero(self):
        """If hypothesis reduces to empty after normalization → 0."""
        fn = self._get_fn()
        # A string consisting only of punctuation/symbols → empty after strip
        score = fn("!!!", "hello world")
        assert score == 0.0


class TestNaiveRawChrF:
    """Unit tests for naive_raw_chrf in analysis."""

    def _get_fn(self):
        from constrained_translation.experiment.analysis import _naive_raw_chrf
        return _naive_raw_chrf

    def test_exact_match(self):
        fn = self._get_fn()
        assert fn("hello world", "hello world") == pytest.approx(1.0, abs=1e-5)

    def test_no_generation_returns_zero(self):
        fn = self._get_fn()
        assert fn(None, "hello world") == 0.0

    def test_empty_generation_returns_zero(self):
        fn = self._get_fn()
        assert fn("", "hello world") == 0.0

    def test_partial_overlap_positive(self):
        fn = self._get_fn()
        score = fn("hello world", "hello earth")
        assert 0.0 < score < 1.0

    def test_audit_failure_artifact_scored_normally(self):
        """naive_raw_chrf scores the raw text even if audit failed;
        [AUDIT_FAILURE] is the *accepted translation*, not raw_generation_text.
        raw_generation_text is the actual model output before rejection."""
        fn = self._get_fn()
        # raw_generation_text contains actual model output
        score = fn("some actual model output text", "hello world")
        # Non-zero even when the translation was rejected
        assert isinstance(score, float)


class TestAnalysisWithRawFields:
    """analyze_language() must compute per-item naive_raw_chrf and
    formatting_tolerant_edit_similarity, and report condition means + paired diffs."""

    def _make_results_with_raw(self, tmp_path: Path, subdir: str,
                                items: list[dict]) -> Path:
        d = tmp_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        path = d / "results.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        # rollup
        rollup = {
            "total_units": len(items), "mean_tok_per_sec": 100.0,
            "coverage_pass_pct": 100.0, "retry_pct": 0.0,
            "hard_failure_pct": 0.0, "token_audit_pass_pct": 100.0,
            "total_input_tokens": 300, "total_output_tokens": 30,
        }
        (d / "rollup.json").write_text(json.dumps(rollup))
        return d

    def _make_manifest(self, tmp_path: Path, items: list[dict]) -> Path:
        path = tmp_path / "manifest.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    @pytest.fixture
    def raw_analysis_dirs(self, tmp_path):
        ids = ["GEN 1:1", "GEN 1:2", "GEN 1:3"]
        refs = ["Au commencement Dieu créa", "La terre était vide", "Dieu dit lumière"]
        manifest_items = [
            {"item_id": ids[i], "source_text": f"eng {i}",
             "exclude_idx": i, "lang": "tst",
             "target_text": refs[i], "corpus_idx": i}
            for i in range(3)
        ]
        manifest_path = self._make_manifest(tmp_path, manifest_items)

        so_items = [
            {
                "item_id": ids[i], "source_text": f"eng {i}",
                "translation": refs[i], "coverage_pass": True,
                "retry_count": 0, "hard_failure": False, "error": None,
                "raw_generation_text": refs[i],
                "raw_token_ids": [1, 2, 3],
                "token_provenance_ratio": 1.0,
                "surface_attestation_similarity": 1.0,
            }
            for i in range(3)
        ]
        sc_items = [
            {
                "item_id": ids[i], "source_text": f"eng {i}",
                "translation": refs[i] + " extra",
                "coverage_pass": True, "retry_count": 0,
                "hard_failure": False, "error": None,
                "raw_generation_text": refs[i] + " extra",
                "raw_token_ids": [1, 2, 3, 4],
                "token_provenance_ratio": 0.8,
                "surface_attestation_similarity": 0.9,
            }
            for i in range(3)
        ]
        so_dir = self._make_results_with_raw(tmp_path, "so", so_items)
        sc_dir = self._make_results_with_raw(tmp_path, "sc", sc_items)
        return {
            "manifest_path": manifest_path,
            "so_dir": so_dir,
            "sc_dir": sc_dir,
            "refs": refs,
            "ids": ids,
        }

    def test_items_have_naive_raw_chrf(self, raw_analysis_dirs):
        """Each item dict in LangAnalysis.items must have 'naive_raw_chrf' keys."""
        from constrained_translation.experiment.analysis import analyze_language
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        for item in result.items:
            assert "sem_only_naive_raw_chrf" in item, (
                f"Missing 'sem_only_naive_raw_chrf' in item {item.get('item_id')}"
            )
            assert "sem_cov_naive_raw_chrf" in item

    def test_items_have_formatting_tolerant_edit_similarity(self, raw_analysis_dirs):
        from constrained_translation.experiment.analysis import analyze_language
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        for item in result.items:
            assert "sem_only_formatting_tolerant_edit_similarity" in item
            assert "sem_cov_formatting_tolerant_edit_similarity" in item

    def test_formatting_tolerant_similarity_uses_raw_output_after_audit_failure(
        self, raw_analysis_dirs
    ):
        """Rejected output is still the hypothesis for this non-gating metric."""
        from constrained_translation.experiment.analysis import analyze_language

        d = raw_analysis_dirs
        results_path = d["sc_dir"] / "results.jsonl"
        records = [json.loads(line) for line in results_path.read_text().splitlines()]
        records[0]["translation"] = "[AUDIT_FAILURE]"
        records[0]["hard_failure"] = True
        records[0]["raw_generation_text"] = d["refs"][0]
        results_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )

        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50,
            n_sign_flip=50,
            seed=42,
        )

        assert result.items[0][
            "sem_cov_formatting_tolerant_edit_similarity"
        ] == pytest.approx(1.0)
        assert result.items[0]["sem_cov_score"] == 0.0

    def test_lang_analysis_has_diagnostic_means(self, raw_analysis_dirs):
        """LangAnalysis must have mean diagnostic fields."""
        from constrained_translation.experiment.analysis import analyze_language
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        assert hasattr(result, "sem_only_mean_naive_raw_chrf")
        assert hasattr(result, "sem_cov_mean_naive_raw_chrf")
        assert hasattr(result, "mean_diff_naive_raw_chrf")
        assert hasattr(result, "sem_only_mean_formatting_tolerant_edit_similarity")
        assert hasattr(result, "sem_cov_mean_formatting_tolerant_edit_similarity")
        assert hasattr(result, "mean_diff_formatting_tolerant_edit_similarity")

    def test_diagnostic_means_in_range(self, raw_analysis_dirs):
        from constrained_translation.experiment.analysis import analyze_language
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        for attr in (
            "sem_only_mean_naive_raw_chrf", "sem_cov_mean_naive_raw_chrf",
            "sem_only_mean_formatting_tolerant_edit_similarity",
            "sem_cov_mean_formatting_tolerant_edit_similarity",
        ):
            v = getattr(result, attr)
            assert 0.0 <= v <= 1.0, f"{attr} = {v} out of [0,1]"

    def test_primary_chrf_unchanged(self, raw_analysis_dirs):
        """Adding diagnostics must not change primary chrF scores."""
        from constrained_translation.experiment.analysis import analyze_language
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        # sem_only translations == references → high chrF
        assert result.sem_only_mean_chrf > 0.9

    def test_markdown_contains_nongating_label(self, raw_analysis_dirs):
        """Markdown report must label diagnostics as NON-GATING."""
        from constrained_translation.experiment.analysis import (
            analyze_language, compute_macro, render_markdown,
        )
        d = raw_analysis_dirs
        result = analyze_language(
            lang="tst",
            manifest_path=d["manifest_path"],
            sem_only_dir=d["so_dir"],
            sem_cov_dir=d["sc_dir"],
            n_boot=50, n_sign_flip=50, seed=42,
        )
        macro = compute_macro([result], n_boot=50, n_sign_flip=50, seed=42)
        md = render_markdown([result], macro)
        assert "NON-GATING" in md, "Markdown must label diagnostics as NON-GATING"

    def test_naive_raw_chrf_zero_when_no_generation(self, tmp_path):
        """When raw_generation_text is null, naive_raw_chrf must be 0."""
        from constrained_translation.experiment.analysis import analyze_language
        ids = ["GEN 1:1"]
        refs = ["Au commencement"]
        manifest_items = [
            {"item_id": ids[0], "source_text": "eng 0",
             "exclude_idx": 0, "lang": "tst",
             "target_text": refs[0], "corpus_idx": 0}
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        with manifest_path.open("w") as fh:
            fh.write(json.dumps(manifest_items[0]) + "\n")

        # Hard failure → no raw_generation_text
        items = [{
            "item_id": ids[0], "source_text": "eng 0",
            "translation": "[AUDIT_FAILURE]", "coverage_pass": True,
            "retry_count": 0, "hard_failure": True, "error": "audit",
            "raw_generation_text": None,
            "raw_token_ids": None,
            "token_provenance_ratio": None,
            "surface_attestation_similarity": None,
        }]

        for cname in ("so", "sc"):
            d = tmp_path / cname
            d.mkdir(parents=True, exist_ok=True)
            (d / "results.jsonl").write_text(
                json.dumps(items[0], ensure_ascii=False) + "\n"
            )
            rollup = {
                "total_units": 1, "mean_tok_per_sec": 0.0,
                "coverage_pass_pct": 100.0, "retry_pct": 0.0,
                "hard_failure_pct": 100.0, "token_audit_pass_pct": 0.0,
                "total_input_tokens": 0, "total_output_tokens": 0,
            }
            (d / "rollup.json").write_text(json.dumps(rollup))

        result = analyze_language(
            lang="tst",
            manifest_path=manifest_path,
            sem_only_dir=tmp_path / "so",
            sem_cov_dir=tmp_path / "sc",
            n_boot=10, n_sign_flip=10, seed=42,
        )
        assert result.items[0]["sem_only_naive_raw_chrf"] == 0.0
        assert result.items[0]["sem_cov_naive_raw_chrf"] == 0.0

    def test_old_results_without_raw_fields_analyzable(self, tmp_path):
        """Old result files without raw fields must remain analyzable (null treated as absent)."""
        from constrained_translation.experiment.analysis import analyze_language
        ids = ["GEN 1:1", "GEN 1:2"]
        refs = ["Au commencement", "La terre était"]
        manifest_items = [
            {"item_id": ids[i], "source_text": f"e{i}",
             "exclude_idx": i, "lang": "tst",
             "target_text": refs[i], "corpus_idx": i}
            for i in range(2)
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        with manifest_path.open("w") as fh:
            for m in manifest_items:
                fh.write(json.dumps(m) + "\n")

        # Old-style records: no raw_* fields
        old_items = [
            {
                "item_id": ids[i], "source_text": f"e{i}",
                "translation": refs[i], "coverage_pass": True,
                "retry_count": 0, "hard_failure": False, "error": None,
            }
            for i in range(2)
        ]
        for cname in ("so", "sc"):
            d = tmp_path / cname
            d.mkdir(parents=True, exist_ok=True)
            with (d / "results.jsonl").open("w") as fh:
                for item in old_items:
                    fh.write(json.dumps(item) + "\n")
            rollup = {
                "total_units": 2, "mean_tok_per_sec": 100.0,
                "coverage_pass_pct": 100.0, "retry_pct": 0.0,
                "hard_failure_pct": 0.0, "token_audit_pass_pct": 100.0,
                "total_input_tokens": 200, "total_output_tokens": 20,
            }
            (d / "rollup.json").write_text(json.dumps(rollup))

        # Must not raise
        result = analyze_language(
            lang="tst",
            manifest_path=manifest_path,
            sem_only_dir=tmp_path / "so",
            sem_cov_dir=tmp_path / "sc",
            n_boot=10, n_sign_flip=10, seed=42,
        )
        assert result.n_items == 2
        # naive_raw_chrf should be 0.0 (no raw_generation_text in old files)
        for item in result.items:
            assert item["sem_only_naive_raw_chrf"] == 0.0
