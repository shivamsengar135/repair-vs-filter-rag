# EXPERIMENT_PLAN.md — preregistration

_Version 1.0, written 2026-11-13, before any experiment on the method has been run._
_Status: **FROZEN — approved by the user on 2026-11-13** (frozen text = git commit `1ea8707`; this status line and
§14 are the only parts edited afterwards). Any change goes in §14 "Deviations" with date, reason and whether
results had already been seen._

Chosen direction (user, 2026-11-13): **Candidate A (Triage-and-Repair before filtering) as the method,
Candidate B as a cost axis, Candidate C as the ablation** — see `METHOD_PROPOSALS.md`.

Availability and licence facts below were checked on 2026-11-13 through the Hugging Face and GitHub APIs
(`logs/phase3_hub_check.log`, `logs/phase3_hub_check_b.log`, `results/phase3/hub_check*.json`).
Items marked **[VERIFY-P4]** could not be confirmed from metadata alone and must be confirmed in Phase 4 before
they are relied on; if one fails, the stated fallback applies and is logged as a deviation.

---

## 1. Research questions and hypotheses

**RQ1 (confirmatory).** At a matched factual-precision target, does repairing low-scoring claims before filtering
keep more correct content than filtering alone?

**RQ2.** What does each stage (decomposition, verification, re-retrieval, revision, filtering) cost, and which
stages pay for themselves?

**RQ3.** How do published mitigation methods compare when run on the same retriever, corpus and open 7–9B
generators?

### Confirmatory hypotheses (the only claims that may be called "significant" in the paper)

- **H1.** On the long-form biography test set (§4), with the primary generator, at target precision 0.90:
  Triage-and-Repair (A) yields more *judge-supported claims per response* than filter-only conformal factuality
  (CF) using the same verifier score.
- **H2.** The same on ASQA.
- **Non-inferiority guard for H1/H2.** A's achieved factual precision on the test set is not lower than CF's by
  more than 2.0 percentage points (one-sided, same bootstrap).

Test: paired bootstrap over questions, 10,000 resamples, two-sided α = 0.05, Holm correction across {H1, H2}.
A hypothesis is "supported" only if the difference is significant **and** its non-inferiority guard holds.
Everything else in the paper is exploratory and will be labelled so.

### Pilot with a pre-set decision rule (first experiment of Phase 5, dev data only)

Measure **R** = among claims that CF at target 0.90 would drop, the fraction for which A's repair step outputs a
claim the judge labels supported. Primary generator, biography dev set + HotpotQA dev set, pooled.

| Outcome | Consequence |
|---|---|
| R ≥ 0.15 | Proceed as planned (method paper). |
| 0.05 ≤ R < 0.15 | Proceed with all planned runs, but frame the paper as a controlled comparative study; A is reported as one arm. |
| R < 0.05 | Same as above, and state plainly that repair did not work in this setup. |

No component of A may be changed in response to the pilot beyond the tuning budget in §9.

---

## 2. Generators

Selection rule: open-weight instruction-tuned, 7–9B, different families, permissive licence, and **must fit in
BF16 on the 20 GB GPU with ≥ 16,000 tokens of KV cache at max-model-len 8192** (measured with vLLM 0.33.0). The
fit test is run in Phase 4 before any experiment; a model that fails is replaced by the listed fallback.

| Role | Model | Params | Licence | Gated | Why |
|---|---|---|---|---|---|
| G1 primary | `Qwen/Qwen2.5-7B-Instruct` @ a09a3545 | 7.62B | Apache-2.0 | no | Fits (measured: 14.29 GiB, 37,920-token cache). Used by VeriCite, SeaRAG, SABER, which makes our numbers comparable in kind. |
| G2 | `mistralai/Mistral-7B-Instruct-v0.3` | 7.25B | Apache-2.0 | no | Different family; smaller than G1 so expected to fit **[VERIFY-P4]**. |
| G3 | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | llama3.1 | **yes (manual)** | Used by Moskvoretskii et al., Rehman et al., SABER. Needs the user to accept the licence and log in. |
| G3 fallback | `ibm-granite/granite-4.2-8b` | 8.79B | Apache-2.0 | no | Newest 8B-class model found (2026-10). 17.6 GB of weights, so the fit test may fail **[VERIFY-P4]**. |

Considered and not chosen: `Qwen/Qwen3-8B` (newer, but its thinking mode adds a confound and token cost);
`mistralai/Ministral-3-8B-Instruct-2512` (8.92B, unlikely to fit in BF16); `google/gemma-2-9b-it` (gated, 9.24B);
Gemma 4 and Qwen 3.5/3.6 (no official 7–9B instruct checkpoint found in the searches).

Decoding: greedy (temperature 0), max 512 new tokens for long-form and 64 for short-form, model's own chat
template, vLLM seed 0. Quantised variants are used only in the generator-size sensitivity study (§8).

**Run tiers (to keep compute bounded).** Tier 1: every method on G1. Tier 2: on G2 and G3 only no-retrieval,
standard RAG, CF, C and A.

---

## 3. Retrieval

- **Corpus:** English Wikipedia, December 2018, 21M passages of 100 words (the DPR `psgs_w100` split of
  Karpukhin et al., which Moskvoretskii et al. state they use; DRAGIN also uses 100-token Wikipedia passages).
  Passage count and file name **[VERIFY-P4]** at download. Source: `facebook/wiki_dpr` (CC-BY-NC-4.0 — research use only;
  to be stated in the paper) or the original DPR download; exact file and checksum recorded at download.
- **Primary retriever:** BM25. Implementation `bm25s` (MIT, pure Python — no Java needed, which matters because
  we cannot install system packages in WSL) with its default parameters, recorded in the config. Fallback if
  bm25s cannot index 21M passages within 62 GB RAM **[VERIFY-P4]**: Pyserini's prebuilt DPR-Wikipedia BM25 index
  (exact index name **[VERIFY-P4]**) with a user-space JDK.
- **Secondary retriever (sensitivity only):** dense, `BAAI/bge-small-en-v1.5` (MIT, 384-dim), passages encoded by
  us, flat inner-product index stored as float16 (~16 GB).
- **k:** top-5 passages to the generator by default; re-retrieval for a claim fetches top-5 for a query built from
  the claim and the question.
- Retrieval results are cached on disk keyed by (retriever, query string, k).

Known mismatch, accepted and stated: biographies are scored against FActScore's own (later) Wikipedia knowledge
base while retrieval uses the 2018 corpus. It affects all methods equally.

---

## 4. Datasets and splits

All subsets are sampled once with seed **20261113**, saved to `data/splits/*.jsonl` with a SHA-256, and never
resampled. "Test" items are touched only in final runs. Split names and sizes of each source are confirmed at
download **[VERIFY-P4]**; if a source has fewer items than a row asks for, all of them are used and this is logged.
(`data/splits/` is small and is committed; `.gitignore` will be adjusted to keep it.)

| Family | Dataset | Source (checked) | Licence | Dev | Test | Why |
|---|---|---|---|---|---|---|
| Short-form | PopQA | `akariasai/PopQA` | not stated on Hub | 500 | 1,000 | Long-tail entities; Self-RAG/CRAG comparison. Single split, so dev and test are disjoint samples of it. |
| Short-form | Natural Questions (open) | `google-research-datasets/nq_open` | CC-BY-SA-3.0 | 500 from train | 1,000 from validation | Standard; used by Moskvoretskii, Rehman. |
| Short-form | TriviaQA | `mandarjoshi/trivia_qa` (`rc.nocontext`) | "unknown" on Hub | 500 from train | 1,000 from validation | Standard; SeaRAG headline dataset. |
| Multi-hop | HotpotQA | `hotpotqa/hotpot_qa` (`fullwiki`) | CC-BY-SA-4.0 | 500 from train | 1,000 from validation | Multi-hop; re-retrieval should matter most here. |
| Multi-hop | 2WikiMultihopQA | `xanhho/2WikiMultihopQA` | Apache-2.0 | 500 from train | 1,000 from dev | Second multi-hop set; FLARE/DRAGIN comparison. |
| Long-form | FActScore biographies | `shmsw25/FActScore` (MIT code) | data licence **[VERIFY-P4]** | see below | see below | Claim-level factual precision; confirmatory H1. |
| Long-form | ASQA (ALCE version) | `princeton-nlp/ALCE-data` | not stated on Hub | 200 | all remaining | Long-form with citation metrics; confirmatory H2. |
| Detection | RAGTruth, re-annotated | `blue-guardrails/ragtruth-plus-plus` (+ original `ParticleMedia/RAGTruth`, MIT) | not stated | — | all 408 | Validates the *verifier component* only (claim vs given context). Not a factuality benchmark and not used to evaluate the full method. |

Biographies: the entity lists and their sizes **[VERIFY-P4]** come from the FActScore release. Rule fixed now: the
human-annotated entities are used only for judge validation (§7); the remaining entities are split 30% dev / 70%
test by the seeded sampler.

HaluEval is not used (largely generated samples; RAGTruth covers the need). MuSiQue is not used (Always-RAG scores
~0.10 on it in Moskvoretskii et al.; floor effects).

---

## 5. Methods

All methods share: corpus, retriever, k, generator, chat template, decoding settings, a 4,096-token prompt budget,
and the evaluation scripts. Prompts are fixed in `configs/prompts/` before test runs.

### 5.1 Our method — Triage-and-Repair (A)

1. **Generate** with standard RAG (top-5).
2. **Decompose** the answer into atomic claims with the generator (few-shot prompt, fixed).
3. **Score** each claim: (i) *support* s = verifier probability that the claim is entailed by any of the retrieved
   passages (max over passages); (ii) *parametric confidence* c = generator's probability of answering "True" to a
   closed-book true/false prompt about the claim (first-token probability, no sampling).
4. **Triage** with thresholds τ_s (support) and τ_c (confidence), tuned on dev:
   - s ≥ τ_s → keep, cite the best-supporting passage;
   - s < τ_s and verifier says *contradicted* → revise the claim from the contradicting passage, re-score;
   - s < τ_s, not contradicted, c ≥ τ_c → targeted re-retrieval (query = question + claim), re-score against the
     new passages; keep if now s ≥ τ_s;
   - otherwise → mark as low.
   At most one repair attempt per claim.
5. **Filter** with a split-conformal threshold on the final score (as in conformal factuality), calibrated on a
   held-out calibration split of pipeline outputs, then **recompose** the surviving claims into an answer; abstain
   if none survive.

Verifier: `lytang/MiniCheck-Flan-T5-Large` (MIT, 3.1 GB) as primary. Its handling of the *contradicted* label
**[VERIFY-P4]**: if MiniCheck is binary (supported / not), contradiction is taken from
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT).

### 5.2 Ablation / Candidate C — two-signal filter without repair

Steps 1–3 and 5 of A; the filter score combines s and c in the FRANQ form
P(faithful)·u_faith + (1 − P(faithful))·u_unfaith. No repair.

### 5.3 Budgeted variant / Candidate B

A with a cap of B repair actions per response, B ∈ {0, 1, 2, 4, ∞}; claims are repaired in order of a cheap risk
score (1 − s). Compared at each B against repairing a uniformly random subset of the low claims of the same size.

### 5.4 Baselines (all run by us)

| Baseline | Implementation | Notes that will be stated in the paper |
|---|---|---|
| No retrieval | ours | |
| Standard RAG, top-5 | ours | |
| Conformal factuality, filter-only (CF) | ours, following Mohri & Hashimoto; score = s | The direct competitor for H1/H2. Their repo (`tatsu-lab/conformal-factual-lm`) has no licence file, so we re-implement from the paper. |
| SelfCheckGPT-NLI filtering | `potsawee/selfcheckgpt` (MIT) | N = 5 sampled answers at temperature 1.0 (N fixed here for cost; the original paper's N is to be noted in the paper **[VERIFY-P4]**); drop sentences above a dev-tuned inconsistency threshold. |
| CoVe (factor + revise) | ours, from the paper's prompts | No official code. Run closed-book as published, and with retrieval-backed verification answers as a labelled variant. |
| RARR-style revise-all | ours, from the paper | Repo has no licence file. Uses our retriever instead of web search — labelled as a modified setup. |
| FLARE (direct) | adapted from `oneal2000/DRAGIN` (Apache-2.0), which implements FLARE for open models | Needs token probabilities; run through vLLM logprobs. |
| Self-RAG | official checkpoint `selfrag/selfrag_llama2_7b` (MIT) with the official inference code (`AkariAsai/self-rag`, MIT) | Its generator is a fine-tuned Llama-2, so it **cannot** share our generators. Reported in its own row group, never as a like-for-like comparison. |

Stretch, only if time allows and clearly labelled as our re-implementations: CRAG without web search (repo has no
licence file; its T5 evaluator weights **[VERIFY-P4]**), DRAGIN (needs attention weights, so runs outside vLLM),
SeaRAG (paper under-specified; see `refs/LIT_REVIEW.md`).

---

## 6. Metrics

| Group | Metric | Definition / tool |
|---|---|---|
| Short-form correctness | Accuracy (match) | A gold answer string appears in the normalised output; normalisation to be aligned with the official Self-RAG evaluation script **[VERIFY-P4]** so the §11 sanity check is like-for-like. Also EM and token F1. |
| Long-form factuality | Factual precision | Fraction of the final answer's atomic claims labelled supported by the judge (§7). |
| | Supported claims per response | Count of judge-supported claims in the final answer. **Primary endpoint of H1/H2.** |
| | Non-empty rate | Fraction of responses with ≥ 1 claim after filtering. |
| | Hallucination rate | 1 − factual precision, and the fraction of responses containing ≥ 1 unsupported claim. |
| Citation (ASQA) | Citation precision / recall | ALCE definitions and scripts (`princeton-nlp/ALCE`, MIT), with the NLI model they prescribe **[VERIFY-P4: it is the 45 GB `google/t5_xxl_true_nli_mixture`; if it cannot run on our GPU in 8-bit, substitute the judge and report this as a deviation]**. |
| Abstention | Coverage, selective accuracy, AURC | Risk–coverage curve from each method's confidence score; AURC by trapezoidal rule. |
| Calibration | ECE | 10 equal-width bins, on answer-level confidence (short-form) and claim-level score (long-form). |
| Cost | Generator calls, prompt tokens, completion tokens, retrieval calls, verifier passes | Counted per question from logs. |
| | GPU-seconds per question | Stage wall-clock ÷ questions, batch processing at concurrency 32, model-load time excluded and reported separately. |
| | Single-stream latency | Measured end to end on a fixed 100-question sample at concurrency 1. |

**Matched-target comparison.** For CF, C and A the filter threshold is calibrated by split conformal prediction on
a calibration split to reach target precision 1 − α ∈ {0.80, 0.85, 0.90, 0.95}; 0.90 is the confirmatory level.
Calibration/test partitioning of the *test set outputs* is repeated over 100 random partitions (seeded 0–99); we
report mean ± std across partitions, and the bootstrap for H1/H2 is nested within partition 0. We report achieved
precision next to the target, and make **no theoretical coverage claim** for A (repair changes the claim
distribution).

---

## 7. Judge and its validation

Long-form factuality needs a claim-level judge. No API models are used.

- **Evidence given to the judge:** passages from the FActScore knowledge base for biographies **[VERIFY-P4:
  availability of their Wikipedia DB]**, and ALCE's gold passages for ASQA. This is a different evidence source
  from what the method sees (our BM25 retrieval), and the judge is a different model from the method's verifier.
- **Judge model:** an instruction-tuned LLM from a family not among the generators under test, chosen on dev
  before any test run. Candidates: `Qwen/Qwen3-14B` in 4-bit **[VERIFY-P4: official AWQ build and fit]**,
  `bespokelabs/Bespoke-MiniCheck-7B` (no licence stated on the Hub — to be checked before use). If only a
  Qwen-family judge fits, G1 results carry a same-family caveat, stated in the paper.
- **Validation 1 — existing human labels:** FActScore's released human annotations. Report the judge's error rate
  and Cohen's κ against them.
- **Validation 2 — our outputs, labelled by the user:** 150 claims sampled (seeded) from A, CF and standard-RAG
  outputs on dev, 50 each, stratified by judge label. The user labels them blind to method and judge label, using
  written guidelines (`docs/LABELLING_GUIDE.md`, to be written in Phase 4). Report Cohen's κ with a bootstrap 95% CI.
- **Acceptance rule (fixed now):** κ ≥ 0.60 on both → judge accepted. 0.40 ≤ κ < 0.60 → results reported with an
  explicit warning and a sensitivity analysis using human labels only on the 150 items. κ < 0.40 → the judge is
  rejected, a second candidate is validated once; if it also fails, long-form conclusions are limited to the
  human-labelled sample.
- Short-form correctness uses string matching, so it needs no judge. A 100-item user check of string-match errors
  is optional.

---

## 8. Ablations and sensitivity (G1, dev for tuning, test for reporting)

Remove or replace one component at a time:
1. no repair (= C); 2. re-retrieval only; 3. revision only; 4. no parametric-confidence axis (one-axis triage);
5. no conformal filter (repair only); 6. verifier replaced by DeBERTa-v3-large NLI, by
`liuyanyi/AlignScore-large-hf`, and by the generator itself (yes/no prompt); 7. decomposition by sentence
splitting instead of LLM.

Sensitivity: top-k ∈ {3, 5, 10}; τ_s over 5 values; retriever BM25 vs dense; generator size — one smaller and one
larger Qwen model in 4-bit **[VERIFY-P4: exact IDs and fit]**; budget B (§5.3).

Failure analysis: 100 seeded errors of A on test (50 wrong-but-kept, 50 correct-but-dropped), categorised by hand
into retrieval miss / decomposition error / verifier false positive / verifier false negative / revision error /
judge error. Examples quoted in the paper come from this sample only.

---

## 9. Tuning budget (identical for every method)

- Tuning uses **dev subsets only**. Test items are never inspected before final runs.
- Each method, baselines included, gets at most **20 dev configurations** on G1; the chosen configuration is
  reused unchanged on G2/G3.
- Every configuration tried is logged in `results/RESULTS_LEDGER.md`, including discarded ones.
- Selection criterion fixed now: short-form — dev accuracy; long-form — supported claims per response at target
  0.90 (methods without a filter: factual precision × claims per response).
- Prompts for all methods are written before tuning and may be edited only as one of the 20 configurations.

---

## 10. Statistics

- Significance level α = 0.05, two-sided, for all tests.
- Paired bootstrap over questions (10,000 resamples) for differences in means; McNemar's test for paired accuracy
  on short-form sets. 95% CIs reported for all headline numbers.
- Multiple comparisons: Holm within the confirmatory family {H1, H2}. Exploratory comparisons are reported with
  unadjusted CIs and labelled exploratory.
- Non-determinism: decoding is greedy. Runs that involve sampling (SelfCheckGPT) or random partitions (conformal
  calibration) use ≥ 3 seeds (sampling: 0, 1, 2) or 100 partitions, and report mean ± std. One G1 standard-RAG run
  is repeated 3 times to measure residual serving non-determinism; that spread is reported.
- No seed, partition or run is dropped for any reason other than a logged technical failure, and a failed run is
  re-run with the same seed.
- "Better" is used only for a significant difference.

---

## 11. Sanity check of the pipeline (Phase 4)

Reproduce one published number in its original setup: **Self-RAG 7B on PopQA, reported 54.9 accuracy**, using the
official checkpoint, code and the retrieval results distributed with the official repo **[VERIFY-P4: that those
files are still downloadable]**. Report our number and the gap whatever it is. Fallback target if the files are
gone: DRAGIN's single-round RAG on HotpotQA with Llama-2-7b-chat (reported EM 0.164) using their code.

---

## 12. Compute plan

Hardware: one RTX 4000 Ada (20 GB), 62 GB RAM visible to WSL, vLLM 0.33.0 in Docker. Measured throughput for G1:
1,256 output tokens/s at 64 concurrent short prompts (`results/RESULTS_LEDGER.md`); RAG-length prompts will be
slower and are to be measured first.

- Models do not fit together (G1 leaves ~1.5 GB). Pipelines therefore run **stage by stage over the whole
  dataset** (generate all → decompose all → score all → repair all → re-score all), swapping models between
  stages, with every stage's outputs cached to disk. Per-question cost is computed from per-stage logs.
- Every stage is resumable (append-only JSONL keyed by question id and config hash) and runs in tmux.
- Rough size of Tier 1: 7 datasets × ~1,000 test questions × ~9 methods × ~4 generator calls ≈ 250,000 calls on
  G1, plus dev tuning of up to 20 configurations on 500-question subsets. I cannot give a trustworthy hour figure
  until RAG-prompt throughput is measured; the Phase 4 gate will report it and, if Tier 1 exceeds ~10 GPU-days,
  propose a reduction for approval **before** any test run.

---

## 13. What will be reported regardless of outcome

- Results for every method, dataset and generator that was run, including those where A loses.
- The pilot's R and which branch of the decision rule was taken.
- All deviations from this plan (§14).
- Numbers from other papers only in a separate table titled "reported in the original paper (different setup)".

---

## 14. Deviations

Each entry: date, what, why, and whether any result of the affected kind had been seen.

**C1 — clarification, 2026-11-13, before any long-form or conformal result existed.**
"Target precision 1 − α" in H1/H2 and §6 was ambiguous. Having re-read Mohri & Hashimoto (arXiv:2402.10978,
Algorithm and Theorem in §3–4), the conformal target is **response-level**: P(all retained claims are correct)
≥ 1 − α. That is what "target 0.90" means for CF, C and A. The primary endpoint (judge-supported claims per
response) and the non-inferiority guard on claim-level factual precision are unchanged. Achieved response-level
full-support rate and claim-level precision are both reported next to the target. Implemented in
`src/hallu/conformal.py`; the guarantee is checked by simulation in `tests/test_conformal.py`.

**C2 — verified facts replacing [VERIFY-P4] tags, 2026-11-13 (no deviation):**
- Corpus: `psgs_w100.tsv.gz` from `dl.fbaipublicfiles.com/dpr/wikipedia_split/`, 4,694,541,059 bytes,
  SHA-256 `c39b020c…53ea8d`, **21,015,324 passages**.
- bm25s indexes the full corpus within RAM: 16 min, peak RSS 46 GB, index 28 GB on disk. Parameters are the
  library defaults k1 = 1.5, b = 0.75, method "lucene"; tokeniser = lower-case, English stop-words, Snowball
  English stemmer; indexed text = title + passage (`~/halluc_data/index/bm25_psgs_w100/build_meta.json`).
- FActScore release: 183 human-annotated entities (three systems, S / NS / IR labels on atomic facts) and 500
  unlabelled entities → 150 dev / 350 test under the 30/70 rule.
- MiniCheck-Flan-T5-Large is binary (supported / not), so contradiction comes from the DeBERTa-v3-large NLI
  model, as the plan's fallback states.
- Dataset split sizes are in `data/splits/MANIFEST.json`; every requested subset size was available.

**D1 — timing method, 2026-11-13, before any timing was reported.** The WSL wall clock was observed jumping
backwards during a run. All durations are therefore measured with a monotonic clock (`time.perf_counter`), not
wall-clock timestamps.

**C3 — details the plan left open, fixed 2026-11-13 before any judged long-form result existed.**
- Candidate C's score: `s + (1 − s)·c`, i.e. P(faithful) = MiniCheck support s, P(true | faithful) taken as 1,
  P(true | unfaithful) = closed-book confidence c (`hallu.pipeline.score_two_signal`).
- Retrieval query: the entity name for biographies, the question text elsewhere. Re-retrieval query: that query
  followed by the claim text.
- Contradiction: NLI contradiction probability ≥ τ_con (default 0.5), maximum over the shown passages; the
  revision prompt receives the passage with the highest contradiction probability.
- A revision that returns "DELETE" removes the claim. A re-retrieved claim keeps the better of its two support
  scores. Method A's filter score is the support score after repair.
- The budgeted variant B is computed offline from A's run: only the B repairs with the lowest original support
  are applied (`hallu.pipeline.final_claims`). Repairs are independent per claim, so no extra model calls are needed.
- Closed-book confidence c that cannot be read from the first token is set to 0.5 and counted in the run summary.

**C4 — judge, fixed 2026-11-13 before any claim was judged.**
- Judge model: `microsoft/phi-4` (MIT, 14.66B; a family not among the generators), served from the
  `RedHatAI/phi-4-FP8-dynamic` build so that it fits in 20 GB **[fit and load still to be confirmed]**. Second
  candidate if it fails validation: `Qwen/Qwen3-14B-AWQ` (same family as G1 — caveat applies).
- Judge prompt: `configs/prompts/judge_v1.yaml`. Label = first-token P("True") ≥ 0.5; unreadable verdicts count as
  not supported and are reported.
- ASQA evidence: ALCE's five oracle-reranked passages plus the human reference long answers.
- Biography evidence: passages of the entity's page in FActScore's knowledge base (blocked until O1 is resolved).

**C5 — matched-target evaluation, fixed 2026-11-13.** Calibration fraction 0.5; partitions seeded 0–99; the same
partitions for every method (`hallu.analysis`). "All retained claims correct" counts an empty answer as correct
(the conformal definition); the non-empty rate is always reported next to it.

**D2 — SelfCheckGPT unit and samples, 2026-11-14, before any SelfCheckGPT score existed.** The baseline scores the
same atomic claims as CF / C / A (the paper scores sentences), and its N = 5 samples are temperature-1.0
re-generations of the same RAG prompt. Scoring replicates the official SelfCheckNLI.predict
(`potsawee/deberta-v3-large-mnli`, mean P(contradiction) over samples).

**D3 — decomposition prompt v2, 2026-11-14, before any judged result.** During bring-up on dev data, prompt v1's
"No claims." escape fired for responses that contain factual content next to statements about the passages:
70 of 200 ASQA dev responses and 6 of 150 biography dev responses had zero claims despite content
(`scripts/audit_decomposition.py`). Prompt set v2 (`configs/prompts/v2.yaml`) changes only `decompose`: it tells the
model to skip statements about the sources but still extract facts, and adds two examples. On ASQA dev, zero-claim
responses fall from 90 to 65 (the rest are mostly answers that only say the passages lack the information); on
biographies from 10 to 9. v2 is used for EVERY method (it is shared measurement infrastructure) and does not count
against any method's tuning budget. The v1 dev runs are kept and marked superseded in the ledger.

**C6 — biography judge evidence, 2026-11-14.** Follows FActScore's own code (`factscore/retrieval.py`,
`factscorer.py`, saved in `refs/raw/code/`): the topic's page from `enwiki-20230401.db` (passages separated by
FActScore's SPECIAL_SEPARATOR), top-5 passages for each claim within that page using FActScore's supported BM25
option (their default GTR-t5-large would add another model), FActScore's prompt layout
("Answer the question about {topic} based on the given context." + passages + "Input: {claim} True or False?").

**C7 — short-form variant of A, 2026-11-14, before any short-form result of A.** A short answer is one claim: the
question and answer are rewritten as one sentence (`configs/prompts/shortform_v1.yaml`, `qa_to_claim`); "revise"
re-answers the question from the contradicting passage; "re-retrieve" re-answers from passages retrieved for
question + claim; a re-retrieved answer replaces the original only if its support score is higher. CF / C / A then
abstain below the calibrated threshold (single-claim conformal filter = selective answering).

**D4 — third generator, 2026-11-16, before any G3 run.** Fit tests (`scripts/fit_test.sh`, results in
`results/phase4/fit/`): Mistral-7B-Instruct-v0.3 PASSES (13.51 GiB, KV cache 30,352 tokens). The G3 fallback
granite-4.2-8b FAILS (16.42 GiB of BF16 weights, 0.12 GiB left for KV cache). Llama-3.1-8B is gated (no access) and
of similar size. New rule, fixed before testing any replacement: fit-test `allenai/Olmo-3-7B-Instruct`
(Apache-2.0, 7.3B, fully open) first, then `tiiuae/Falcon3-7B-Instruct` (TII Falcon license); G3 = the first that
passes the unchanged ≥16,000-token rule. If neither passes, G3 is dropped and the paper reports two generators.
*Outcome (2026-11-16):* Olmo-3-7B-Instruct FAILS (13.59 GiB weights; its KV cache needs ~0.5 MB/token, so only
~6,200 tokens fit). Falcon3-7B-Instruct @ 1e57a0ec PASSES (13.93 GiB, 25,344 KV tokens, 23.0 tok/s single stream).
**G3 = tiiuae/Falcon3-7B-Instruct.** Licence: TII Falcon-LLM License 2.0 (Dec 2024; Apache-2.0-based, perpetual
royalty-free grant, subject to TII's Acceptable Use Policy), read on falconllm.tii.ae on 2026-11-16.
The judge (Phi-4 FP8) is not subject to the generator rule; it runs with max-model-len 4096 at GPU util 0.92.

**D5 — Phase 5 execution decisions, 2026-11-16, before any test run (user approved the Phase 4 gate).**
- Pilot branch (§1): R = 0.042 → the paper is framed as a controlled comparative study; H1/H2 are still tested
  exactly as preregistered and reported whatever the outcome.
- Tuning: every method keeps its single default configuration (configuration #1 everywhere — the papers' defaults
  for baselines, τ = 0.5 for CF / C / A). This uses the same budget (1 of 20) for every method; no dev tuning.
- Test runs start before judge validation 2 (the user's 150 labels). If that validation rejects the judge, the
  preregistered fallback applies (second judge, re-judge) — generations are unaffected.
- Short-form baselines are no-retrieval, standard RAG, CF / C / A and Self-RAG (own row group); CoVe, RARR-style
  and FLARE run on the long-form sets only; SelfCheckGPT runs on long-form with 3 sampling seed sets (§10).
- Scheduling only (no effect on results): long-form verifier scores are computed on the GPU between generator
  sessions (`--stop-after`), and serving non-determinism is measured by 3 cache-free repeats of PopQA test standard RAG.

**D6 — Phase 5 incidents and Phase 6 set-up, 2026-11-17 (after all Phase 5 test runs except Self-RAG; chosen by the
assistant under the user's standing instruction to pick defaults and continue through gates).**
- Environment only: `TORCH_DISABLE_NATIVE_JIT=1` (torch 2.14 native Triton kernels need a C compiler, absent in WSL);
  stock torch/cuBLAS kernels are used instead. torch / transformers versions unchanged. Short-form verifier scores
  were computed on CPU for all three generators (consistent), long-form scores on GPU for all three (consistent).
- §8 generator size: smaller = `Qwen/Qwen2.5-3B-Instruct` (bf16, same precision as G1, rev aa8e7253…), larger =
  `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit AWQ, rev 53953585…; 4-bit is required to fit, so size and quantisation are
  confounded for the larger model — to be stated in the paper). Both subject to the §2 fit test.
- §8 verifier swap to AlignScore uses the HF conversion `liuyanyi/AlignScore-large-hf` (rev cc13dc57…, MIT,
  base_model yzha/AlignScore) loaded by our own code (`hallu.verifier.AlignScore`; remote code not executed);
  scoring follows the shipped AlignScore inference code (tri-label head, P(aligned)). Equivalence to the original
  checkpoint is not independently verified — reported as such.
- §3 dense retriever: `BAAI/bge-small-en-v1.5` (rev 5c38ec7c…), exact inner-product search over all 21,015,324
  passages (no ANN approximation). Per the model card (read 2026-11-17): CLS pooling, L2-normalised embeddings,
  queries prefixed with "Represent this sentence for searching relevant passages: ", passages without a prefix.
- All model downloads for Phase 6 were done in one batch on 2026-11-17 at the user's request.
- §8 scope on TEST, G1: ablations 1 (= C) and 5 (no filter) come from the main runs; 2 (re-retrieval only) and
  3 (revision only) are computed from the main run's stored repairs (the branches act on disjoint claims and are
  computed independently, so this equals re-running with one branch off); 4 one-axis triage = c ignored, every
  unsupported non-contradicted claim is re-retrieved; 6 verifier = DeBERTa-NLI P(entailment) / AlignScore /
  generator P(True | passage) (prompt `configs/prompts/ablation_v1.yaml`, unreadable verdict → 0); 7 = NLTK Punkt
  sentences. Sensitivity: k ∈ {3, 10}, τ_s ∈ {0.3, 0.4, 0.6, 0.7}, dense retriever, generator size. These run on the
  two long-form sets (the claim-level setting); generator size also runs on the five short-form sets.
  Orchestration: `scripts/phase6_runs.sh` (log `logs/phase6_runs.log`). AWQ model served with dtype float16.

**Defaults chosen by the assistant at the user's instruction ("choose the default decision by yourself"),
2026-11-13:** G3 = `ibm-granite/granite-4.2-8b` (no Llama licence was granted), subject to the fit test;
BM25 stays the primary retriever; a single human annotator is assumed for judge validation.

**Open problem O1 (not yet a deviation), 2026-11-13.** FActScore's knowledge base (`enwiki-20230401.db`, 21.9 GB)
is hosted on Google Drive; the download stopped at 9.2 GB with Google's "too many users have viewed or downloaded
this file recently" quota error. Partial file kept; retry later or download via a browser.

---

## 15. Open decisions for the user (needed before or during Phase 4)

1. **Llama-3.1-8B access.** Accept Meta's licence on the Hub and run `hf auth login` in WSL, or approve the
   Granite fallback for G3.
2. **Second human annotator.** One labeller gives judge-vs-human κ but no human-vs-human agreement. A second
   person labelling the same 150 claims would let us report it.
3. **Non-commercial corpus licence** (`facebook/wiki_dpr`, CC-BY-NC-4.0) — fine for research; confirm it suits
   the target journal's data policy once the journal is chosen.
4. **Target journal** — still open; not needed until Phase 7.
5. **A C compiler in WSL** is still missing. It is not needed for this plan as written (bm25s, vLLM in Docker), but
   any dependency that compiles native code will fail.
