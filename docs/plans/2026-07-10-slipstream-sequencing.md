# Slipstream Sequencing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Measure how arbitrary, silver, and golden source-only acquisition ordering changes whole-project coverage and fixed-eval constrained translation per human-translated cell.

**Architecture:** Add a deterministic sequencing layer above the existing constrained translation pipeline. Evaluate source coverage for every eligible remaining-project cell each round, but call vLLM only for supported cells; persist immutable manifests and append-only per-cell/per-round records, then render a graph bundle after each language-policy-seed permutation.

**Tech Stack:** Python 3.11, pytest, existing BM25/coverage retrieval, vLLM 0.19, XGrammar 0.1.33, Qwen3.5-9B, matplotlib, JSONL/JSON/CSV.

---

## Frozen definitions

- Shared source indices and normalized-source equivalence groups are used across `mya`, `npi`, `ckb`, and `tpi`.
- Pools are disjoint after equivalence-group expansion: seed (default 10), acquisition (40), fixed eval (60), and remaining project (all other eligible rows).
- Round 0 is seed-only. Round `r` for 1–40 is measured after acquisition `r` is added.
- Selection is source-only. Target references are revealed only after a candidate has been selected.
- Silver maximizes current source support fraction, then BM25 similarity to the translated pool, then stable corpus index.
- Golden maximizes the sum of remaining-project occurrence counts for newly covered normalized source types, then unweighted new types, then stable corpus index.
- `predicted_supported` means no unsupported normalized source unit under the current translated pool.
- `realized_accepted` means generation occurred and all backend-control, exact-ID, full-decode, surface-attestation, and model-generated-UNK audits passed.
- Source abstention artifact: detector marks a unit unknown although the canonical unit is present in the translated source pool. This is a normalization/index defect.
- Constraint artifact: a source-supported generation is rejected despite its visible lexical surface being licensed, due only to tokenization, byte decoding, terminal control, grammar, or normalization handling.
- Emitted hallucination: visible or token-ID lexical output is unlicensed. Escaped hallucination: such output is accepted or serialized as translation; this must be zero.
- `licensed_failure_zero_chrf` uses all source-supported eval cells as denominator and gives rejected generations zero. Accepted-only chrF is diagnostic. Abstentions are excluded from licensed-region quality but retained in project-completion effectiveness.
- Golden accounting calibration compares ex-ante weighted gain with deterministic ex-post source-coverage reduction. Generalization evidence is the non-circular fixed-eval accepted/quality change.

## Artifact contract

Each run lives at `experiments/slipstream/<manifest_id>/<language>/<policy>/<seed>/` and contains:

- `config.json`, including git commit, model, tokenizer, policy, random seed, pool sizes, normalization version, and manifest digest.
- `selection.jsonl`, one acquisition decision per round with candidate scores and ex-ante gain.
- `cells.jsonl`, one record per evaluated cell/round/view with source provenance, coverage units, unknown spans, retrieval IDs, grammar digest, raw generation, exact IDs, audit result/category, references, and quality metrics.
- `rounds.csv` and `rounds.jsonl`, one metric snapshot per round.
- `summary.json`, AUC, early slope, cells-to-target, safety totals, calibration errors, and counts.
- `graphs/`, the required PNG graph bundle.
- `README.md`, generated data dictionary and command needed to reproduce the permutation.

Never overwrite or resume across a changed manifest, code revision, normalization version, or metric schema.

---

### Task 1: Build leakage-safe sequencing manifests

**Files:**
- Create: `constrained_translation/experiment/sequencing_manifest.py`
- Test: `tests/constrained_translation/experiment/test_sequencing_manifest.py`

**Steps:** Write failing tests for shared indices, four disjoint pools, normalized-equivalence exclusion, marker filtering, deterministic seed, and manifest digest. Run focused tests to verify RED. Implement the minimum manifest builder and serializer. Verify focused and full offline suites. Commit.

### Task 2: Implement corpus coverage accounting

**Files:**
- Create: `constrained_translation/experiment/corpus_coverage.py`
- Test: `tests/constrained_translation/experiment/test_corpus_coverage.py`

**Steps:** Test canonical type/occurrence counts, unsupported spans, fully-supported cells, weighted/unweighted remaining, false source abstention detection, marginal gain, and exact ex-ante/ex-post accounting. Verify RED, implement, verify GREEN/full suite, commit.

### Task 3: Implement acquisition policies

**Files:**
- Create: `constrained_translation/experiment/policies.py`
- Test: `tests/constrained_translation/experiment/test_policies.py`

**Steps:** Test ≥5 deterministic arbitrary permutations, silver support/similarity ordering, golden occurrence-weighted gain, source-only candidate inputs, stable tie-breaking, and no access to target/reference fields. Verify RED, implement, verify GREEN/full suite, commit.

### Task 4: Define safety and constraint-cost taxonomy

**Files:**
- Create: `constrained_translation/experiment/outcomes.py`
- Modify only if required: `constrained_translation/token_auditor.py`, `protocol.py`
- Test: `tests/constrained_translation/experiment/test_outcomes.py`

**Steps:** Test safe source abstention, false source abstention, accepted licensed generation, constraint artifact, emitted hallucination, and escaped-hallucination invariant with distinct sentinel fields. Verify RED, implement without weakening grammar/audits, verify GREEN/full suite, commit.

### Task 5: Implement the round runner and durable records

**Files:**
- Create: `constrained_translation/experiment/sequencing_runner.py`
- Test: `tests/constrained_translation/experiment/test_sequencing_runner.py`

**Steps:** Test round 0, reveal-after-selection, project/fixed views, coverage gate before backend call, append-only records, resume validation, raw rejected-output retention, and per-round metrics. Use `FakeBackend`; verify RED, implement, verify GREEN/full suite, commit.

### Task 6: Implement metrics and calibration

**Files:**
- Create: `constrained_translation/experiment/sequencing_analysis.py`
- Test: `tests/constrained_translation/experiment/test_sequencing_analysis.py`

**Steps:** Test weighted/unweighted AUC orientation, early slope, cells-to-50/70/90, arbitrary variance inputs, licensed failure-zero chrF, accepted-only diagnostics, predicted-support vs realized-acceptance reliability bins/ECE, golden accounting error, and fixed-eval realized gains. Verify RED/GREEN/full suite, commit.

### Task 7: Generate the per-permutation graph bundle

**Files:**
- Create: `constrained_translation/experiment/sequencing_plots.py`
- Test: `tests/constrained_translation/experiment/test_sequencing_plots.py`

**Steps:** Test all required graph filenames, nonempty PNGs, correct project/fixed labels, denominator labels, and arbitrary variance-band support. Implement headless matplotlib rendering and generated README/data dictionary. Verify tests, commit.

### Task 8: Add CLI and pre-flight gates

**Files:**
- Modify: `constrained_translation/experiment/cli.py`
- Test: `tests/constrained_translation/experiment/test_sequencing_cli.py`

**Steps:** Add `sequence-prepare`, `sequence-run`, and `sequence-analyze` commands. Test model endpoint, manifest/config compatibility, language list, disjoint pools, output freshness, and refusal to loosen constraints. Verify RED/GREEN/full suite, commit.

### Task 9: Offline integration and two-stage review

Run the complete offline suite with no skips hidden. Execute one four-round fake-backend permutation and inspect success and rejection paths end-to-end. Perform spec-compliance review, then code-quality review; fix and re-review. Commit and push.

### Task 10: GPU pilot gate

Start Qwen3.5-9B on vLLM/XGrammar and run one language × one policy × four acquisitions against the full project coverage gate plus a small fixed eval. Verify terminal controls, RTL/Unicode, byte decode, leakage guards, artifact taxonomy, emitted/escaped hallucinations, graph/data completeness, and supported-cell model-call count. Abort scale-up if false source abstention, escaped hallucination, stale resume, or uncontrolled GPU workload occurs.

### Task 11: Full permutations with immediate delivery

Run each `(language, policy, seed)` independently: five arbitrary seeds, one silver, one golden for each language. After each run, verify record counts and graph files, then send its graph bundle and concise interpretation immediately. After five arbitrary seeds for a language, generate its variance band. Finally compute four-language crossed language × item bootstrap summaries without treating cells as independent.
