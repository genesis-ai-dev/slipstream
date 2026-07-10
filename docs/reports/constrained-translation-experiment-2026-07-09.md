# Constrained Few-Shot Translation Experiment

Date: 2026-07-09  
Repository: `genesis-ai-dev/ebibletools`  
Branch: `feat/constrained-translation-loop`  
Model: self-hosted `Qwen/Qwen3.5-9B`  
Server: vLLM 0.19.0, tensor parallel 2 × RTX 4090  
Structured-output backend: XGrammar

## Executive summary

A strict constrained few-shot translation loop was implemented and tested across English→Myanmar (`mya`), Nepali (`npi`), Central Kurdish (`ckb`), Somali (`som`), Tok Pisin (`tpi`), and Tongan (`ton`). The experiment compared semantic-only retrieval (10 examples) with semantic+coverage retrieval (5 semantic and up to 5 coverage-targeted examples, followed by logged evidence-expansion retries).

On the final leakage-clean paired sample of 60 held-out source units per language:

- Source coverage rose from 25.0% to 80.0%.
- Strictly accepted units rose from 63/360 (17.5%) to 149/360 (41.4%).
- Strict hard-failure rate fell from 82.5% to 58.6%.
- Strict failure-zero chrF+ rose from 0.0760 to 0.1360, a paired macro difference of +0.0600.
- Non-gating raw chrF+ rose from 0.1054 to 0.2911, a paired difference of +0.1857.
- Formatting-tolerant edit similarity rose from 0.1128 to 0.3073, a paired difference of +0.1944.

The raw and formatting-tolerant gains generalized directionally across all six languages and remained positive under crossed language×item bootstrap analysis. The strict effect was heterogeneous: Somali's strict score declined despite a large raw-quality gain because many Somali generations violated exact tokenizer-ID provenance. Therefore coverage retrieval clearly improves evidence availability and reference similarity, but the original all-or-nothing token-ID criterion remains operationally costly and language-dependent.

## Invariants preserved

- No unconstrained generation fallback.
- Retries only expand evidence and remain grammar-constrained.
- `[UNK:source_span]` is inserted deterministically outside model generation.
- Every accepted lexical token ID has example provenance.
- Every accepted surface lexical unit is attested in the selected target examples.
- Grammar operates over attested target surface units, not arbitrary subword permutations.
- Exact generated tokenizer IDs are audited after decoding.
- Sentence-aligned co-occurrence is described only as sentence-level evidence, never as word alignment.
- Target-vocabulary de-bias pruning remains disabled without word alignments or gloss annotations.
- Audit and backend failures remain hard failures.
- Non-gating scalar diagnostics never alter strict acceptance.

## Implementation

The implementation lives under `constrained_translation/` and includes:

- deterministic two-stage `ExampleSelector`;
- source-only inverted index for coverage retrieval;
- held-out source-equivalence-group exclusion;
- `VocabExtractor` with sentence-level provenance;
- shared source-unit normalization and `UNKDetector`;
- per-request XGrammar EBNF construction;
- strict prompt builder;
- vLLM 0.19 structured-output HTTP backend;
- exact token-ID and full-sequence BPE/byte-aware surface auditor;
- final-position-only Qwen `<|im_end|>` control handling;
- append-only JSONL event/provenance logging;
- batch rollups;
- deterministic manifests, paired runner, resumability, and analysis;
- raw chrF+, formatting-tolerant edit similarity, token provenance ratio, and surface-attestation similarity diagnostics.

## Evaluation hardening discovered during execution

Three defects were found through live multilingual testing and corrected before the final run:

1. **Qwen terminal control token**: vLLM includes the final `<|im_end|>` ID in generated token IDs while omitting it from visible text. It is now treated as protocol control only when it occurs exactly once in final position. Interior or repeated occurrences fail loudly.
2. **Equivalent-source leakage**: excluding only the held-out corpus index allowed repeated verses to retrieve identical source text through another row. Retrieval now excludes the complete exact normalized source-equivalence group from both stages.
3. **Corpus markers**: structural `<range>` rows could enter manifests. Explicit corpus markers are now filtered across all target languages before shared sampling.

The final run used a new output directory and newly generated manifests. Its 60 held-out source texts were unique and contained no `<range>` references.

## Experimental design

### Conditions

- Semantic-only: `n_semantic=10`, `n_coverage=0`, no retry.
- Semantic+coverage: `n_semantic=5`, `n_coverage=5`, maximum two coverage-expansion retries.

### Shared settings

- `temperature=0`
- `max_tokens=128`
- seed `20260709`
- 60 identical held-out source indices per target language
- 360 language-item pairs and 720 condition-level records
- hard failures score zero in the primary strict quality metric

### Metrics

**Strict/gating**

- normalized source coverage;
- strict token and surface audit;
- strict accepted rate;
- hard-failure rate;
- failure-zero chrF+.

**Non-gating diagnostics**

- naive raw chrF+ against the reference even if strict audit rejects the generation;
- formatting-tolerant character edit similarity after NFKC normalization, removal of Unicode format/punctuation/symbol characters, and whitespace collapse;
- exact token provenance ratio among generated lexical token occurrences;
- surface-attestation similarity, computed from generated lexical units against their closest attested surface units.

## Aggregate operational results

### Semantic-only

- Total units: 360
- Source coverage passed: 90/360 (25.0%)
- Generated after coverage: 90
- Strictly accepted: 63/360 (17.5%)
- Strict audit pass among generated units: 70.0%
- Hard failures: 297/360 (82.5%)
- Retried: 0
- Input tokens: 107,721
- Output tokens: 5,481

### Semantic+coverage

- Total units: 360
- Source coverage passed: 288/360 (80.0%)
- Generated after coverage: 288
- Strictly accepted: 149/360 (41.4%)
- Strict audit pass among generated units: 51.7%
- Hard failures: 211/360 (58.6%)
- Retried: 78/360 (21.7%)
- Input tokens: 268,712
- Output tokens: 20,819
- Accepted-output throughput by language: approximately 70.5–77.1 generated tokens/second

Semantic+coverage generated many more units, but its audit pass rate conditional on generation was lower. The richer output exposed more possible exact-ID and surface-composition violations. Nevertheless, its total strict accepted rate more than doubled.

## Semantic+coverage failure decomposition

Across 360 language-item pairs:

- Accepted: 149 (41.4%)
- No source coverage after retry: 72 (20.0%)
- Unlicensed lexical token ID: 103 (28.6%)
- Unattested surface recombination: 36 (10.0%)

By language, strict accepted counts were:

- Myanmar: 20/60
- Nepali: 37/60
- Central Kurdish: 35/60
- Somali: 6/60
- Tok Pisin: 31/60
- Tongan: 20/60

Somali was the major strict-audit outlier: raw output quality was strong, but 39/60 records failed on unlicensed IDs.

## Quality results by language

Values are semantic+coverage minus semantic-only.

- Myanmar: strict +0.0249; raw diagnostic approximately +0.0856; formatting-tolerant approximately +0.1447.
- Nepali: strict +0.0838; raw approximately +0.1303; formatting-tolerant approximately +0.1512.
- Central Kurdish: strict +0.1349; raw approximately +0.1926; formatting-tolerant approximately +0.2115.
- Somali: strict -0.0429; raw approximately +0.2292; formatting-tolerant approximately +0.2153.
- Tok Pisin: strict +0.0820; raw approximately +0.1988; formatting-tolerant approximately +0.1802.
- Tongan: strict +0.0772; raw approximately +0.2775; formatting-tolerant approximately +0.2637.

The exact per-item records and machine-generated summaries are in `experiments/measured-n60-clean/analysis/`.

## Statistical analysis

Because the same 60 source units were translated into six target languages, 360 observations are not independent. Results were therefore inspected three ways:

1. target-language means as six clusters;
2. source-item means as 60 clusters;
3. crossed bootstrap resampling both target languages and source items.

### Strict failure-zero chrF+

- Mean paired difference: +0.0600
- Language-bootstrap 95% CI: approximately [+0.0110, +0.0994]
- Exact language-level sign-flip p: 0.09375
- Item-cluster bootstrap 95% CI: approximately [+0.0235, +0.0946]
- Item-cluster sign-flip p: approximately 0.0019
- Crossed language×item bootstrap 95% CI: approximately [-0.0024, +0.1210]

Interpretation: the strict aggregate effect is positive, but it is not robustly homogeneous across target languages. The crossed interval includes zero, driven primarily by Somali. The pooled item-level p-value in the standard report (`0.0001`) should not be read as if all language-item observations were independent.

### Naive raw chrF+ — non-gating

- Mean paired difference: +0.1857
- Language-bootstrap 95% CI: approximately [+0.1348, +0.2341]
- Exact language-level sign-flip p: 0.03125
- Item-cluster bootstrap 95% CI: approximately [+0.1411, +0.2313]
- Item-cluster sign-flip Monte Carlo p: <0.00001
- Crossed language×item bootstrap 95% CI: approximately [+0.1210, +0.2606]
- Positive/zero/negative source-item clusters: 42/12/6

### Formatting-tolerant edit similarity — non-gating

- Mean paired difference: +0.1944
- Language-bootstrap 95% CI: approximately [+0.1635, +0.2278]
- Exact language-level sign-flip p: 0.03125
- Item-cluster bootstrap 95% CI: approximately [+0.1456, +0.2437]
- Item-cluster sign-flip Monte Carlo p: <0.00001
- Crossed language×item bootstrap 95% CI: approximately [+0.1368, +0.2594]
- Positive/zero/negative source-item clusters: 39/12/9

Only one raw semantic+coverage output exactly matched its reference. Removing every item with an exact reference match left the diagnostic effects effectively unchanged.

## Interpretation

1. **Coverage-driven retrieval works.** It increased normalized source coverage by 55 percentage points and raised strict accepted output from 17.5% to 41.4%.
2. **The quality improvement is not merely formatting.** Both raw chrF+ and formatting-tolerant edit similarity show large positive effects with crossed confidence intervals excluding zero.
3. **Boolean strictness hides useful gradation.** Semantic+coverage outputs averaged approximately 0.872 exact token provenance and 0.884 surface-attestation similarity among generated units, even though many failed the all-or-nothing gate.
4. **Exact token-ID provenance is tokenizer-sensitive.** A surface-valid attested word may be encoded with a context-dependent token ID absent from the example tokenizations. This is especially costly in Somali and Tongan.
5. **Strict effectiveness is heterogeneous.** Five languages improved on strict failure-zero chrF+, while Somali declined. A pooled strict result alone would overstate generality.
6. **The method is useful as a conservative translation-assistance signal, not yet as an autonomous translator.** Twenty percent of coverage-condition items still lacked source evidence after retries, and another 38.6% generated text that failed strict provenance or surface composition.

## Limitations

- eBible rows are sentence-aligned translations, not word alignments.
- Source-unit overlap is a coverage proxy, not proof that every source sense has a corresponding target rendering.
- chrF+ and edit similarity measure reference overlap, not theological or semantic correctness.
- The six target languages are a purposive sample, not a random sample of all low-resource languages.
- Language-cluster inference has only six clusters; the smallest possible nonzero exact two-sided all-positive sign-flip p-value is 0.03125.
- Qwen may contain Bible text from pretraining. The final evaluation prevented corpus-row and exact-source-equivalence retrieval leakage, but it cannot remove model pretraining memory.
- Dynamic grammars constrain surface strings; they do not force the tokenizer to choose the same tokenization path observed in examples. Post-generation exact-ID audit is therefore stricter than XGrammar surface acceptance.

## Reproduction commands

Prepare clean manifests:

```bash
python -m constrained_translation.experiment prepare \
  --corpus-dir Corpus \
  --vref benchmarks/data/vref.txt \
  --output-dir experiments/measured-n60-clean \
  --languages mya npi ckb som tpi ton \
  --n 60 \
  --seed 20260709
```

Run paired conditions:

```bash
python -m constrained_translation.experiment run \
  --corpus-dir Corpus \
  --output-dir experiments/measured-n60-clean \
  --languages mya npi ckb som tpi ton \
  --seed 20260709 \
  --vllm-url http://127.0.0.1:8000 \
  --model Qwen3.5-9B \
  --coverage-max-retries 2 \
  --max-tokens 128 \
  --temperature 0
```

Analyze:

```bash
python -m constrained_translation.experiment analyze \
  --output-dir experiments/measured-n60-clean \
  --languages mya npi ckb som tpi ton \
  --seed 20260709
```

Run tests:

```bash
python -m pytest -q
```

## Artifacts

- Final local experiment: `experiments/measured-n60-clean/`
- Standard report: `experiments/measured-n60-clean/analysis/summary.md`
- Per-language JSON: `experiments/measured-n60-clean/analysis/<lang>_analysis.json`
- Cluster-aware diagnostics: `experiments/measured-n60-clean/analysis/clustered_diagnostics.json`
- Implementation report: `docs/reports/constrained-translation-experiment-2026-07-09.md`
