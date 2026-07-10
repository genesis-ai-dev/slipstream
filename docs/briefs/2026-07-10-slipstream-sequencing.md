# Experiment Brief: Slipstream Sequencing for Constrained Batch Translation

Owner: Ryder Wishart · Context: Aquilla batch-translation R&D · Date: 2026-07-10
Predecessor: clean six-language constrained-translation run (2026-07-09)

---

## 1. What this experiment is actually for

Aquilla already predicts any single cell with few-shot in-context learning, which is why it works for ultra-low-resource, out-of-training languages — our core differentiator. This experiment tests how to add large-batch translation on top of that without losing two properties:

1. it still uses few-shot prompting (no fine-tuning), and
2. it fails loudly on unlearned material instead of hallucinating.

### The slipstream frame (read this before touching metrics)

The batch predictor is always running — the water always flows. The human never "does the translation"; they place banks (expert-translated cells) that redirect the flow. Grammar-constrained decoding is the bank wall: the model may only emit wordings attested in the retrieved few-shot evidence. An optional agent tells the human where to place the next bank for maximum redirection.

Therefore this is not primarily a translation-quality study. It is an effort-efficiency study. The quantity we are selling is *redirection per unit of human input*: how much of the remaining project gets correctly channeled per cell a human hand-translates. The x-axis of every learning curve is human effort.

---

## 2. Headline question

> Given what has been translated so far, does coverage-guided sequencing let a human place fewer banks to redirect more water than arbitrary order — and can we trust the system's own coverage numbers enough to point a translator at the next batch?

Success does not require high chrF. Success requires:
- (A) Effort efficiency: coverage-guided policies reduce unknown corpus material (occurrence-weighted) faster per human cell than random.
- (B) Calibration: when the system says X% of the remaining corpus is supported, ~X% is actually accepted/translatable.
- (C) Safety: hallucination rate ≈ 0 by construction; failures are safe abstentions, not fabrications.

---

## 3. Setup

- Languages: four, drawn from the prior clean run, chosen to span scripts/morphology (e.g. Myanmar, Nepali, Central Kurdish, Tok Pisin). Keep the same normalization/tokenization fixes from 2026-07-09 (im_end token, RTL marks, <range> markers, byte-level decoding, repeated-source leakage guard).
- Model: Qwen3.5-9B, few-shot / in-context only. No fine-tuning, no added tokens.
- Serving: vLLM + XGrammar constrained decoding. Per-request grammar compiled from the licensed token set. Not OpenAI-style static logit_bias.
- Per language, three disjoint pools:
  - Acquisition set: 40 sentences, representing the project the human will translate over time.
  - Remaining project: the rest of the corpus, predicted each round (project-completion view).
  - Fixed external eval set: held out, identical across all policies and rounds (generalization view). Never enters any few-shot pool.
- Seed: the same small translated seed pool for every policy so round 0 is identical.

---

## 4. Policies (with leakage discipline)

All selection must use only signals available before the target is translated — source-side similarity and source-side coverage against the current pool. No policy may peek at the reference, post-hoc chrF, or any target-side alignment.

- Arbitrary (baseline): random acquisition order. Run several seeds (≥5) to produce a real variance band, not one permutation.
- Silver — easiest next: add the acquisition sentence best supported by / most similar to the current translated pool (source-side). Expected to give the steepest *early* gains by deepening an already-covered region.
- Golden — max marginal coverage: add the sentence that most reduces frequency-weighted unknown source occurrences across the remaining corpus. Expected to broaden evidence and dominate later. (It is the agent's real "where to place the next bank" recommendation.)
> Golden is *defined* by corpus coverage, so scoring golden by corpus coverage is circular and proves nothing. The honest test of whether coverage buys quality is the fixed external eval set — material no policy selected. Make the fixed-eval curve the generalization headline; treat the project-completion curve as the operational (partly self-fulfilling) view.

---

## 5. Per-round procedure

For each language, each policy, each round (0 → 40):

1. Select the next acquisition sentence per policy; add its reference translation to the translated pool (simulates the human placing a bank).
2. Retrieve few-shot examples from the pool (semantic + coverage-targeted, per prior method).
3. Deterministically compute unsupported source spans before generation and mark them [UNK:source].
4. Build the XGrammar grammar from the licensed token set (union of target tokens in retrieved examples).
5. Predict (a) every remaining project segment and (b) the fixed external eval set under constraint.
6. Record all metrics in §6.

---

## 6. Metrics

### 6.1 Primary — effort efficiency (the headline)
- Weighted unknown source occurrences remaining vs. round (primary curve).
- Unweighted unknown source types remaining vs. round (so function words don't disguise poor lexical breadth — report both).
- Cells-to-target: human cells required to reach coverage thresholds (e.g. 50/70/90% supported).
- Gain per cell: marginal reduction in weighted unknown occurrences from each added sentence.
- AUC of the 40-round curve; early-stage slope.

### 6.2 Safety — abstention vs. hallucination (must be scored separately)
Split every failure into:
- Safe abstention — output was [UNK] because there was genuinely no source evidence. This is a success of the guarantee; it is the system pointing at where a bank is needed. Do not score it as a quality miss.
- Hallucination — model emitted an unlicensed wording (unlicensed lexical token ID, or unattested surface recombination). This must sit ≈ 0. Report hallucination rate as a first-class safety number.

Quality (§6.4) is measured only within the licensed region — i.e., conditional on the system having chosen to translate rather than abstain.

### 6.3 Constraint cost — the leak that manufactures effort
Decompose every [UNK] into:
- (a) Genuine no-evidence → human input truly required (correct behavior).
- (b) Constraint/tokenizer artifact → a genuinely attested form was rejected by grammar/BPE/byte handling (a false alarm that would trigger an unnecessary human query).

Report the (b) rate as a cost line against the coverage benefit. High (b) means the dam is leaking effort; it is an engineering fault, not a property of the language. (Prior run: 28.6% unlicensed-token-ID + 10% unattested recombination — quantify how much of that is (b).)

### 6.4 Supporting quality (not the objective)
- Failure-zero chrF+ as the continuous curve (primary quality signal).
- Provenance ratio and surface-attestation similarity as continuous diagnostics.
- Strict acceptance kept only as a conservative secondary bound — the Boolean gate is brittle (Somali tokenizer-ID problem) and will add noise over 40 rounds; do not let it be the headline.

### 6.5 Calibration & trust (currently missing — add explicitly)
- Calibration curve: predicted % supported vs. realized % accepted/translatable, across rounds and languages (reliability diagram). This is what proves the meter can be trusted.
- Realized marginal gain check: when golden estimates "adding sentence X removes Y% of unknown occurrences," compare ex-ante Y against ex-post realized reduction. This proves the agent's *recommendations* are trustworthy, not just its measurements.

---

## 7. Analysis & statistics
- Per-language curves plus a four-language aggregate.
- Crossed language × item clustered analysis (as in the prior run); bootstrap CIs.
- Arbitrary reported as a variance band over its seeds.
- Two complementary curves kept distinct throughout: project-completion (operational) and fixed-eval (generalization headline).
- Interpret shape (slope, AUC, ranking, cells-to-target) rather than small absolute chrF deltas — n is small and chrF is noisy.

---

## 8. Success criteria (restated as pass/fail)
1. Effort: golden and/or silver reach coverage/quality targets in fewer human cells than arbitrary, on the fixed-eval curve, with the ordering visible in AUC and cells-to-target.
2. Calibration: predicted-supported tracks realized-accepted within a stated tolerance.
3. Safety: hallucination rate ≈ 0; the dominant failure mode is safe abstention.
4. Constraint cost: false-abstention rate (b) is measured and acceptably low (or identified as a fixable engineering leak).

The experiment is a success even if chrF stays modest, provided 1–4 hold.

---

## 9. Deliverables per permutation
Generate and share after each permutation completes (do not wait for the full multilingual run):
- Weighted + unweighted unknown-remaining curves.
- Fully-predictable remaining segments by round.
- Fixed-eval performance by round.
- Failure-zero chrF and safe-abstention vs. hallucination decomposition by round.
- Constraint-cost (b-rate) by round.
- Calibration curve and realized-vs-estimated marginal gain.
- Per-sample marginal gain; cumulative learning-curve area.
- All underlying data saved and documented alongside the graphs.

---

## 10. Explicitly out of scope
- Fine-tuning or adding tokens to the model (in-context only — this is what keeps it working for languages with only a handful of pairs).
- Any output not traceable to attested few-shot evidence, even if it "looks" plausible.
- Optimizing throughput by loosening the constraint. Throughput is secondary; the constraint is the product.
- Claiming a source unknown maps to a specific target word — without alignment we have co-occurrence, not lexical alignment. All unknown/coverage claims are source-side.
