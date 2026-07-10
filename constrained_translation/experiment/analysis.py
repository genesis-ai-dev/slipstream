"""constrained_translation.experiment.analysis — Metric analysis.

Computes per-language and macro aggregate statistics for paired
semantic-only vs semantic+coverage runs.

Metrics
-------
* chrF+ (from metrics.chrf.chrF_plus, signature: hypothesis, reference,
  n_char=6, n_word=2, beta=2.0, remove_punctuation=True)
* System metrics from rollup: coverage_pass_pct, hard_failure_pct,
  retry_pct, token_audit_pass_pct, mean_tok_per_sec

Statistical analysis (stdlib + numpy only, no scipy/sacrebleu)
--------------------------------------------------------------
* Bootstrap 95% CI for mean difference (B=2000 resamples, seed-controlled)
* Paired sign-flip randomization p-value (R=10000 iterations, seed-controlled)
* Report heterogeneity via per-language SDs rather than overclaiming
  a single macro effect.

Output files
------------
<output_dir>/analysis/
  <lang>_analysis.json        — per-language full results
  macro_analysis.json         — macro aggregate
  summary.md                  — Markdown report
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from metrics.chrf import chrF_plus


# ---------------------------------------------------------------------------
# Non-gating diagnostic helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[lb]


def _normalize_for_edit(text: str) -> str:
    """NFKC, remove Cf / P* / S* characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    buf = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            # Keep only printable non-Cf; Cf = Format chars
            if cat == "Cf":
                continue
            # Other C categories (Cc, Cs, Co, Cn): skip
            continue
        if cat.startswith("P") or cat.startswith("S"):
            continue
        buf.append(ch)
    collapsed = " ".join("".join(buf).split())
    return collapsed


def _formatting_tolerant_edit_similarity(hypothesis: str, reference: str) -> float:
    """1-normalised character edit distance after aggressive normalization.

    Returns 0.0 if either side normalizes to empty.
    """
    h = _normalize_for_edit(hypothesis)
    r = _normalize_for_edit(reference)
    if not h or not r:
        return 0.0
    max_len = max(len(h), len(r))
    return 1.0 - _levenshtein(h, r) / max_len


def _naive_raw_chrf(raw_generation_text: Optional[str], reference: str) -> float:
    """chrF+ of raw generation text vs reference, regardless of audit outcome.

    Returns 0.0 when raw_generation_text is None or empty.
    """
    if not raw_generation_text or not reference:
        return 0.0
    return chrF_plus(hypothesis=raw_generation_text, reference=reference)


# ---------------------------------------------------------------------------
# Bootstrap CI and sign-flip p-value (stdlib + numpy)
# ---------------------------------------------------------------------------

def _bootstrap_mean_diff_ci(
    diffs: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for mean of *diffs*."""
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, dtype=float)
    n = len(arr)
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def _sign_flip_p_value(
    diffs: list[float],
    n_iter: int = 10000,
    seed: int = 42,
) -> float:
    """Paired sign-flip randomization test p-value (two-sided).

    H0: condition labels are exchangeable (no systematic difference).
    Statistic: absolute mean difference.
    """
    rng = random.Random(seed)
    arr = [float(d) for d in diffs]
    observed = abs(sum(arr) / len(arr))
    n = len(arr)
    count_extreme = 0
    for _ in range(n_iter):
        flipped = [d * (1 if rng.random() > 0.5 else -1) for d in arr]
        stat = abs(sum(flipped) / n)
        if stat >= observed:
            count_extreme += 1
    return (count_extreme + 1) / (n_iter + 1)  # +1 for continuity correction


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_results(path: str | Path) -> list[dict]:
    """Load a results.jsonl file."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _load_rollup(path: str | Path) -> dict:
    """Load a rollup.json file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_manifest(path: str | Path) -> list[dict]:
    """Load manifest JSONL."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Per-item chrF+ scorer
# ---------------------------------------------------------------------------

def _score_item(translation: str, reference: str) -> float:
    """Compute chrF+ for one item; return 0.0 on empty/failure."""
    if not translation or not reference:
        return 0.0
    # Skip failure artifacts
    if "[COVERAGE_FAILURE]" in translation or "[AUDIT_FAILURE]" in translation:
        return 0.0
    return chrF_plus(hypothesis=translation, reference=reference)


# ---------------------------------------------------------------------------
# Per-language analysis
# ---------------------------------------------------------------------------

@dataclass
class ItemScore:
    item_id: str
    sem_only_score: float
    sem_cov_score: float
    diff: float          # sem_cov - sem_only
    reference: str
    sem_only_translation: str
    sem_cov_translation: str
    # Non-gating diagnostics
    sem_only_naive_raw_chrf: float = 0.0
    sem_cov_naive_raw_chrf: float = 0.0
    sem_only_formatting_tolerant_edit_similarity: float = 0.0
    sem_cov_formatting_tolerant_edit_similarity: float = 0.0


@dataclass
class LangAnalysis:
    lang: str
    n_items: int
    sem_only_mean_chrf: float
    sem_cov_mean_chrf: float
    mean_diff: float
    diff_ci_lo: float
    diff_ci_hi: float
    sign_flip_p: float
    # System metrics (from rollup)
    sem_only_coverage_pass_pct: float
    sem_cov_coverage_pass_pct: float
    sem_only_hard_failure_pct: float
    sem_cov_hard_failure_pct: float
    sem_only_retry_pct: float
    sem_cov_retry_pct: float
    # Per-item details
    items: list[dict]
    # Audit and throughput health (from rollup)
    sem_only_token_audit_pass_pct: float = 0.0
    sem_cov_token_audit_pass_pct: float = 0.0
    sem_only_mean_tok_per_sec: float = 0.0
    sem_cov_mean_tok_per_sec: float = 0.0
    # NON-GATING DIAGNOSTIC means
    sem_only_mean_naive_raw_chrf: float = 0.0
    sem_cov_mean_naive_raw_chrf: float = 0.0
    mean_diff_naive_raw_chrf: float = 0.0
    sem_only_mean_formatting_tolerant_edit_similarity: float = 0.0
    sem_cov_mean_formatting_tolerant_edit_similarity: float = 0.0
    mean_diff_formatting_tolerant_edit_similarity: float = 0.0


def analyze_language(
    lang: str,
    manifest_path: str | Path,
    sem_only_dir: str | Path,
    sem_cov_dir: str | Path,
    n_boot: int = 2000,
    n_sign_flip: int = 10000,
    seed: int = 42,
) -> LangAnalysis:
    """Compute analysis for one language.

    Parameters
    ----------
    lang:
        Language code.
    manifest_path:
        Path to the manifest JSONL (has target_text for references).
    sem_only_dir:
        Directory with semantic_only results/rollup.
    sem_cov_dir:
        Directory with semantic_coverage results/rollup.
    n_boot, n_sign_flip, seed:
        Bootstrap and randomization parameters.
    """
    manifest_items = _load_manifest(manifest_path)
    sem_only_results = _load_results(Path(sem_only_dir) / "results.jsonl")
    sem_cov_results = _load_results(Path(sem_cov_dir) / "results.jsonl")
    sem_only_rollup = _load_rollup(Path(sem_only_dir) / "rollup.json")
    sem_cov_rollup = _load_rollup(Path(sem_cov_dir) / "rollup.json")

    # Index by item_id
    sem_only_by_id = {r["item_id"]: r for r in sem_only_results}
    sem_cov_by_id = {r["item_id"]: r for r in sem_cov_results}

    item_scores: list[ItemScore] = []
    for m in manifest_items:
        iid = m["item_id"]
        ref = m["target_text"]
        so = sem_only_by_id.get(iid, {})
        sc = sem_cov_by_id.get(iid, {})
        so_trans = so.get("translation", "")
        sc_trans = sc.get("translation", "")
        # A loud system failure is not a translation hypothesis. Score it as
        # zero regardless of the deterministic diagnostic artifact stored in
        # ``translation`` (for example source text containing [UNK:…] spans).
        so_score = 0.0 if so.get("hard_failure", True) else _score_item(so_trans, ref)
        sc_score = 0.0 if sc.get("hard_failure", True) else _score_item(sc_trans, ref)

        # Non-gating diagnostics
        so_raw = so.get("raw_generation_text", None)
        sc_raw = sc.get("raw_generation_text", None)
        so_naive_raw = _naive_raw_chrf(so_raw, ref)
        sc_naive_raw = _naive_raw_chrf(sc_raw, ref)
        so_ftes = _formatting_tolerant_edit_similarity(so_raw or "", ref)
        sc_ftes = _formatting_tolerant_edit_similarity(sc_raw or "", ref)

        item_scores.append(ItemScore(
            item_id=iid,
            sem_only_score=so_score,
            sem_cov_score=sc_score,
            diff=sc_score - so_score,
            reference=ref,
            sem_only_translation=so_trans,
            sem_cov_translation=sc_trans,
            sem_only_naive_raw_chrf=so_naive_raw,
            sem_cov_naive_raw_chrf=sc_naive_raw,
            sem_only_formatting_tolerant_edit_similarity=so_ftes,
            sem_cov_formatting_tolerant_edit_similarity=sc_ftes,
        ))

    diffs = [s.diff for s in item_scores]
    so_scores = [s.sem_only_score for s in item_scores]
    sc_scores = [s.sem_cov_score for s in item_scores]

    mean_diff = float(np.mean(diffs))
    ci_lo, ci_hi = _bootstrap_mean_diff_ci(diffs, n_boot=n_boot, alpha=0.05, seed=seed)
    p_val = _sign_flip_p_value(diffs, n_iter=n_sign_flip, seed=seed)

    # Non-gating diagnostic means
    so_raw_chrfs = [s.sem_only_naive_raw_chrf for s in item_scores]
    sc_raw_chrfs = [s.sem_cov_naive_raw_chrf for s in item_scores]
    so_ftes_vals = [s.sem_only_formatting_tolerant_edit_similarity for s in item_scores]
    sc_ftes_vals = [s.sem_cov_formatting_tolerant_edit_similarity for s in item_scores]

    return LangAnalysis(
        lang=lang,
        n_items=len(item_scores),
        sem_only_mean_chrf=float(np.mean(so_scores)),
        sem_cov_mean_chrf=float(np.mean(sc_scores)),
        mean_diff=mean_diff,
        diff_ci_lo=ci_lo,
        diff_ci_hi=ci_hi,
        sign_flip_p=p_val,
        sem_only_coverage_pass_pct=sem_only_rollup.get("coverage_pass_pct", 0.0),
        sem_cov_coverage_pass_pct=sem_cov_rollup.get("coverage_pass_pct", 0.0),
        sem_only_hard_failure_pct=sem_only_rollup.get("hard_failure_pct", 0.0),
        sem_cov_hard_failure_pct=sem_cov_rollup.get("hard_failure_pct", 0.0),
        sem_only_retry_pct=sem_only_rollup.get("retry_pct", 0.0),
        sem_cov_retry_pct=sem_cov_rollup.get("retry_pct", 0.0),
        items=[asdict(s) for s in item_scores],
        sem_only_token_audit_pass_pct=sem_only_rollup.get("token_audit_pass_pct", 0.0),
        sem_cov_token_audit_pass_pct=sem_cov_rollup.get("token_audit_pass_pct", 0.0),
        sem_only_mean_tok_per_sec=sem_only_rollup.get("mean_tok_per_sec", 0.0),
        sem_cov_mean_tok_per_sec=sem_cov_rollup.get("mean_tok_per_sec", 0.0),
        sem_only_mean_naive_raw_chrf=float(np.mean(so_raw_chrfs)),
        sem_cov_mean_naive_raw_chrf=float(np.mean(sc_raw_chrfs)),
        mean_diff_naive_raw_chrf=float(np.mean(sc_raw_chrfs)) - float(np.mean(so_raw_chrfs)),
        sem_only_mean_formatting_tolerant_edit_similarity=float(np.mean(so_ftes_vals)),
        sem_cov_mean_formatting_tolerant_edit_similarity=float(np.mean(sc_ftes_vals)),
        mean_diff_formatting_tolerant_edit_similarity=(
            float(np.mean(sc_ftes_vals)) - float(np.mean(so_ftes_vals))
        ),
    )


# ---------------------------------------------------------------------------
# Macro analysis
# ---------------------------------------------------------------------------

@dataclass
class MacroAnalysis:
    n_languages: int
    languages: list[str]
    macro_sem_only_chrf: float
    macro_sem_cov_chrf: float
    macro_mean_diff: float
    # Heterogeneity: SD of per-language mean differences
    lang_diff_sd: float
    # Per-language diffs (for heterogeneity reporting)
    per_lang_mean_diffs: dict[str, float]
    per_lang_sem_only_chrf: dict[str, float]
    per_lang_sem_cov_chrf: dict[str, float]
    # Pooled bootstrap CI and sign-flip p (all items pooled)
    pooled_diff_ci_lo: float
    pooled_diff_ci_hi: float
    pooled_sign_flip_p: float


def compute_macro(
    lang_analyses: list[LangAnalysis],
    n_boot: int = 2000,
    n_sign_flip: int = 10000,
    seed: int = 42,
) -> MacroAnalysis:
    """Compute macro-level statistics across all languages."""
    langs = [a.lang for a in lang_analyses]
    per_lang_diffs = {a.lang: a.mean_diff for a in lang_analyses}
    per_lang_so = {a.lang: a.sem_only_mean_chrf for a in lang_analyses}
    per_lang_sc = {a.lang: a.sem_cov_mean_chrf for a in lang_analyses}

    diff_vals = list(per_lang_diffs.values())
    so_vals = list(per_lang_so.values())
    sc_vals = list(per_lang_sc.values())

    lang_diff_sd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0

    # Pool all item-level diffs
    all_diffs = []
    for a in lang_analyses:
        all_diffs.extend(item["diff"] for item in a.items)

    ci_lo, ci_hi = _bootstrap_mean_diff_ci(all_diffs, n_boot=n_boot, seed=seed)
    p_val = _sign_flip_p_value(all_diffs, n_iter=n_sign_flip, seed=seed)

    return MacroAnalysis(
        n_languages=len(langs),
        languages=langs,
        macro_sem_only_chrf=float(np.mean(so_vals)),
        macro_sem_cov_chrf=float(np.mean(sc_vals)),
        macro_mean_diff=float(np.mean(diff_vals)),
        lang_diff_sd=lang_diff_sd,
        per_lang_mean_diffs=per_lang_diffs,
        per_lang_sem_only_chrf=per_lang_so,
        per_lang_sem_cov_chrf=per_lang_sc,
        pooled_diff_ci_lo=ci_lo,
        pooled_diff_ci_hi=ci_hi,
        pooled_sign_flip_p=p_val,
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def render_markdown(
    lang_analyses: list[LangAnalysis],
    macro: MacroAnalysis,
) -> str:
    """Render a Markdown summary report."""
    lines = [
        "# Multilingual Experiment: Semantic-Only vs Semantic+Coverage",
        "",
        "## Per-Language chrF+ Results",
        "",
    ]

    for a in lang_analyses:
        lines += [
            f"### Language: `{a.lang}` (n={a.n_items})",
            "",
            f"- **Semantic-only** mean chrF+: {_fmt(a.sem_only_mean_chrf)}",
            f"- **Semantic+Coverage** mean chrF+: {_fmt(a.sem_cov_mean_chrf)}",
            f"- Mean difference (cov − sem): {_fmt(a.mean_diff)} "
            f"[95% CI: {_fmt(a.diff_ci_lo)}, {_fmt(a.diff_ci_hi)}]",
            f"- Sign-flip p-value: {_fmt(a.sign_flip_p, 4)}",
            "",
            "**System metrics**",
            "",
            f"- Coverage pass %: sem-only={_fmt(a.sem_only_coverage_pass_pct, 1)}"
            f", sem+cov={_fmt(a.sem_cov_coverage_pass_pct, 1)}",
            f"- Hard failure %: sem-only={_fmt(a.sem_only_hard_failure_pct, 1)}"
            f", sem+cov={_fmt(a.sem_cov_hard_failure_pct, 1)}",
            f"- Retry %: sem-only={_fmt(a.sem_only_retry_pct, 1)}"
            f", sem+cov={_fmt(a.sem_cov_retry_pct, 1)}",
            f"- Token audit pass %: sem-only={_fmt(a.sem_only_token_audit_pass_pct, 1)}"
            f", sem+cov={_fmt(a.sem_cov_token_audit_pass_pct, 1)}",
            f"- Mean generated tokens/sec: sem-only={_fmt(a.sem_only_mean_tok_per_sec, 1)}"
            f", sem+cov={_fmt(a.sem_cov_mean_tok_per_sec, 1)}",
            "",
            "**NON-GATING DIAGNOSTIC metrics** (do not affect acceptance)",
            "",
            f"- Naive raw chrF+ (sem-only): {_fmt(a.sem_only_mean_naive_raw_chrf)}",
            f"- Naive raw chrF+ (sem+cov):  {_fmt(a.sem_cov_mean_naive_raw_chrf)}",
            f"- Naive raw chrF+ paired diff: {_fmt(a.mean_diff_naive_raw_chrf)}",
            f"- Formatting-tolerant edit similarity (sem-only):"
            f" {_fmt(a.sem_only_mean_formatting_tolerant_edit_similarity)}",
            f"- Formatting-tolerant edit similarity (sem+cov):"
            f"  {_fmt(a.sem_cov_mean_formatting_tolerant_edit_similarity)}",
            f"- Formatting-tolerant edit similarity paired diff:"
            f" {_fmt(a.mean_diff_formatting_tolerant_edit_similarity)}",
            "",
        ]

    lines += [
        "## Macro Summary",
        "",
        f"- Languages: {', '.join(f'`{l}`' for l in macro.languages)}",
        f"- Macro mean sem-only chrF+: {_fmt(macro.macro_sem_only_chrf)}",
        f"- Macro mean sem+cov chrF+: {_fmt(macro.macro_sem_cov_chrf)}",
        f"- Macro mean diff: {_fmt(macro.macro_mean_diff)} "
        f"[pooled 95% CI: {_fmt(macro.pooled_diff_ci_lo)}, {_fmt(macro.pooled_diff_ci_hi)}]",
        f"- Pooled sign-flip p: {_fmt(macro.pooled_sign_flip_p, 4)}",
        f"- Per-language diff SD (heterogeneity): {_fmt(macro.lang_diff_sd)}",
        "",
        "**Interpretation note**: The pooled test must be read alongside the",
        "per-language effects and heterogeneity; it does not establish a",
        "homogeneous treatment effect across languages.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def run_analysis(
    output_dir: str | Path,
    languages: list[str],
    n_boot: int = 2000,
    n_sign_flip: int = 10000,
    seed: int = 42,
) -> tuple[list[LangAnalysis], MacroAnalysis]:
    """Run full analysis over all languages.

    Expects the standard output directory layout:
        <output_dir>/<lang>/<lang>_manifest.jsonl
        <output_dir>/<lang>/semantic_only/results.jsonl + rollup.json
        <output_dir>/<lang>/semantic_coverage/results.jsonl + rollup.json

    Returns (lang_analyses, macro_analysis) and writes JSON + Markdown to
        <output_dir>/analysis/
    """
    base = Path(output_dir)
    analysis_dir = base / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lang_analyses: list[LangAnalysis] = []
    for lang in languages:
        lang_dir = base / lang
        manifest_path = lang_dir / f"{lang}_manifest.jsonl"
        sem_only_dir = lang_dir / "semantic_only"
        sem_cov_dir = lang_dir / "semantic_coverage"

        analysis = analyze_language(
            lang=lang,
            manifest_path=manifest_path,
            sem_only_dir=sem_only_dir,
            sem_cov_dir=sem_cov_dir,
            n_boot=n_boot,
            n_sign_flip=n_sign_flip,
            seed=seed,
        )
        lang_analyses.append(analysis)

        # Write per-language JSON
        lang_json_path = analysis_dir / f"{lang}_analysis.json"
        with lang_json_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(analysis), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    macro = compute_macro(lang_analyses, n_boot=n_boot, n_sign_flip=n_sign_flip, seed=seed)

    # Write macro JSON
    with (analysis_dir / "macro_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(macro), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Write Markdown summary
    md = render_markdown(lang_analyses, macro)
    with (analysis_dir / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write(md)

    return lang_analyses, macro
