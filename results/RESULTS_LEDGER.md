# Results ledger

Every number that appears in the paper must have a row here. No row → the number does not go in the paper.
Every run is logged, including failed and superseded ones (mark them, never delete them).

## Paper numbers (table cell / figure → provenance)

| Paper item | Value | Script | Config | Seed(s) | Git commit | Output file | Status |
|---|---|---|---|---|---|---|---|
| Setup: test-set sizes | PopQA, NQ, TriviaQA, HotpotQA, 2Wiki 1,000 each; biographies 350; ASQA 748 (dev 500 / 500 / 500 / 500 / 500 / 150 / 200) | `scripts/build_splits.py`, `scripts/build_longform_splits.py` (Phase 4) | — | 20261113-* | see MANIFEST sha256 | `data/splits/MANIFEST.json` | final |
| Setup: corpus | 21,015,324 passages (DPR psgs_w100) | download + `scripts/build_bm25.py` | bm25s k1 = 1.5, b = 0.75 | — | — | EXPERIMENT_PLAN.md §14 C2; `~/halluc_data/index/bm25_psgs_w100/build_meta.json` | final |
| Every result table / figure of the paper | — | see the "TEST runs (Phase 5)" and "Phase 6 analyses" sections below | | | | `results/test/tables_phase5.md`, `tables_ablations.md`, `cost_table.md`, `results/figures/test/` | draft (Phase 7) |

## Infrastructure measurements (NOT paper results)

| Item | Value | Script | Settings | Git commit | Output file | Date |
|---|---|---|---|---|---|---|
| vLLM throughput, Qwen2.5-7B-Instruct BF16, concurrency 1 | 22.4 out tok/s (8 requests) | `scripts/run_phase0_bench.sh` → `scripts/bench_vllm.py` | vLLM 0.33.0 Docker image, `scripts/serve_vllm_docker.sh` defaults (max-model-len 8192, gpu-mem-util 0.90, seed 0); max_tokens 256, temperature 0; single run, no repeats | 1eab75a (serve_vllm_docker.sh + run_phase0_bench.sh committed afterwards) | `results/phase0/vllm_bench_qwen2.5-7b-instruct_bf16.json`, `logs/phase0_bench.log` | 2026-11-12 |
| same, concurrency 8 | 174.9 out tok/s total; 21.9 per request (16 requests) | same | same | same | same | 2026-11-12 |
| same, concurrency 32 | 665.1 out tok/s total; 20.9 per request (64 requests) | same | same | same | same | 2026-11-12 |
| same, concurrency 64 | 1256.1 out tok/s total; 19.7 per request (128 requests) | same | same | same | same | 2026-11-12 |
| VRAM for the above | weights 14.29 GiB; 18317 MiB in use with server up (incl. ~720 MiB desktop); KV cache 37,920 tokens | — | same | — | `logs/vllm_docker_20261112_114143.log` | 2026-11-12 |

Caveats: one run per level (no variance estimate); short prompts (~40 tokens) with a 73.5% prefix-cache hit rate, so
RAG-length prompts will be slower; cross-check — server log reported 1253.1 gen tok/s at 64 concurrent.

## Dev-set runs (NOT paper results — pipeline bring-up and tuning only)

Every dev configuration counts against the 20-configuration budget of its method (EXPERIMENT_PLAN.md §9).
Each row's numbers are in `<output dir>/summary.json`, which also stores the config, its hash and the git commit.

| Date | Method | Generator | Dataset (dev, n=500) | Match acc. | EM | F1 | Config # for this method | Output dir |
|---|---|---|---|---|---|---|---|---|
| 2026-11-13 | closed-book, prompts v1 | Qwen2.5-7B-Instruct | PopQA | 0.174 | 0.160 | 0.189 | 1 | `results/dev/popqa/closed_book__Qwen2.5-7B-Instruct__f1de76e36917/` |
| 2026-11-13 | closed-book, prompts v1 | Qwen2.5-7B-Instruct | NQ | 0.192 | 0.134 | 0.220 | 1 | `results/dev/nq/closed_book__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | closed-book, prompts v1 | Qwen2.5-7B-Instruct | TriviaQA | 0.518 | 0.458 | 0.531 | 1 | `results/dev/triviaqa/closed_book__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | closed-book, prompts v1 | Qwen2.5-7B-Instruct | HotpotQA | 0.226 | 0.206 | 0.286 | 1 | `results/dev/hotpotqa/closed_book__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | closed-book, prompts v1 | Qwen2.5-7B-Instruct | 2WikiMultihopQA | 0.384 | 0.374 | 0.398 | 1 | `results/dev/2wiki/closed_book__Qwen2.5-7B-Instruct__*/` |

| 2026-11-13 | standard RAG (BM25 top-5), prompts v1 | Qwen2.5-7B-Instruct | PopQA | 0.312 | 0.256 | 0.305 | 1 | `results/dev/popqa/rag__Qwen2.5-7B-Instruct__ea900975ad2e/` |
| 2026-11-13 | standard RAG (BM25 top-5), prompts v1 | Qwen2.5-7B-Instruct | NQ | 0.318 | 0.260 | 0.356 | 1 | `results/dev/nq/rag__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | standard RAG (BM25 top-5), prompts v1 | Qwen2.5-7B-Instruct | TriviaQA | 0.652 | 0.542 | 0.633 | 1 | `results/dev/triviaqa/rag__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | standard RAG (BM25 top-5), prompts v1 | Qwen2.5-7B-Instruct | HotpotQA | 0.444 | 0.392 | 0.483 | 1 | `results/dev/hotpotqa/rag__Qwen2.5-7B-Instruct__*/` |
| 2026-11-13 | standard RAG (BM25 top-5), prompts v1 | Qwen2.5-7B-Instruct | 2WikiMultihopQA | 0.386 | 0.372 | 0.394 | 1 | `results/dev/2wiki/rag__Qwen2.5-7B-Instruct__*/` |

Script for all rows: `scripts/run_qa.py` (via `scripts/run_qa_all.sh <method> dev Qwen/Qwen2.5-7B-Instruct`), greedy
decoding, seed 0. Closed-book rows: git commit 248bd13, log `logs/phase4_closed_book_dev.log`. RAG rows: git commit
4888a70, log `logs/phase4_rag_dev.log`.

Retrieval diagnostic on the same dev sets (fraction of questions with a gold answer string in the top-5 BM25
passages; not meaningful for yes/no questions): PopQA 0.390, NQ 0.464, TriviaQA 0.728, HotpotQA 0.544, 2Wiki 0.428.

Notes on these numbers:
- The NQ and TriviaQA dev subsets come from the *train* splits (single-annotator answers, e.g. gold "five" vs
  prediction "5"), so dev accuracy is not comparable with test-set numbers in the literature.
- On 2Wiki, RAG (0.386) is no better than closed-book (0.384) with single-shot BM25 retrieval.
- PopQA retrieval is weak because many subjects have short ambiguous titles ("The Hit", "Contract"); the subject's
  own page is in the top-5 for 26% of questions (`scripts/inspect_retrieval.py`).
- Cost on PopQA dev, cold cache: RAG used 840 prompt tokens per question; 500 generations took 76.5 s at
  concurrency 32; BM25 retrieval took 0.6 s per query (numpy backend, memory-mapped index, not yet optimised).

### Long-form dev runs (G1 = Qwen2.5-7B-Instruct, prompts v1) — pipeline statistics, NOT judged

Counts come from each run's `summary.json`. "Supported" here means MiniCheck support ≥ 0.5 against the retrieved
passages — the method's own verifier, not the judge. Judged factuality comes later.

| Run | Dataset | n | Key statistics | Config # | Output dir |
|---|---|---|---|---|---|
| Triage-and-Repair pipeline (A / C / CF / standard RAG from one run), τ_s = τ_c = τ_con = 0.5 | bio dev | 150 | 19.1 claims/response; 10 responses with no claims; 8 answers truncated at 512 tokens. Actions: keep 2,506, revise 174, re-retrieve 88, low 96. Revision: 169 of 174 returned DELETE, 2 became supported. Re-retrieval: 26 of 88 became supported. Verifier on CPU: MiniCheck 1,863 s, NLI 1,787 s. | A:1, C:1, CF:1 | `results/dev/bio/longform__Qwen2.5-7B-Instruct__6a6082d9435e/` |
| CoVe factor+revise, closed-book | bio dev | 150 | 7.9 planned facts/response; verdicts consistent 363 / partial 263 / inconsistent 555; 5.0 final claims/response; 9 empty answers | 1 | `results/dev/bio/cove-closed_book__Qwen2.5-7B-Instruct__6f7c2d1c8889/` |
| CoVe factor+revise, retrieval-backed verification | bio dev | 150 | verdicts 226 / 301 / 654; 4.4 final claims/response; 17 empty | 1 | `results/dev/bio/cove-rag_verify__*/` |
| CoVe factor+revise, closed-book | ASQA dev | 200 | 3.7 planned facts; verdicts 402 / 112 / 219; 1.7 final claims; 28 empty | 1 | `results/dev/asqa/cove-closed_book__*/` |
| CoVe factor+revise, retrieval-backed verification | ASQA dev | 200 | verdicts 251 / 228 / 254; 1.4 final claims; 39 empty | 1 | `results/dev/asqa/cove-rag_verify__*/` |
| RARR-style revise-all | bio dev | 150 | 7.4 questions/response; 200 disagreements of 1,107 checks; 170 edits in 101 responses; 14.0 final claims | 1 | `results/dev/bio/rarr__*/` |
| FLARE-direct, θ = 0.8, β = 0.4 (paper values) | bio dev | 150 | 8.0 sentences/response; retrieval on 99.5% of look-ahead sentences; 9.2 final claims | 1 | `results/dev/bio/flare__*/` |

Logs: `logs/phase4_longform_dev.log`, `logs/phase4_cove_dev.log`, `logs/phase4_baselines_dev.log`.
**The rows above use prompts v1 and are SUPERSEDED by the v2 rows below (deviation D3).**

### Long-form dev runs, prompts v2 (G1) — pipeline statistics, NOT judged

| Run | Dataset | n | Key statistics | Output dir |
|---|---|---|---|---|
| Triage-and-Repair pipeline (A / C / CF / standard RAG), τ = 0.5 | ASQA dev | 200 | 2.2 claims/response; 65 responses with zero claims; keep 373, revise 27, re-retrieve 26, low 17; all 27 revisions returned DELETE; 9 of 26 re-retrievals became supported | `results/dev/asqa/longform__Qwen2.5-7B-Instruct__33f74812f8a7/` |
| CoVe closed-book | ASQA dev | 200 | 2.26 final claims/response; 28 empty | `results/dev/asqa/cove-closed_book__Qwen2.5-7B-Instruct__515f55fc1215/` |
| CoVe retrieval-verified | ASQA dev | 200 | 2.09 final claims; 39 empty | `results/dev/asqa/cove-rag_verify__Qwen2.5-7B-Instruct__375c900c29f7/` |
| RARR-style | ASQA dev | 200 | 4.52 final claims; 174 edits | `results/dev/asqa/rarr__Qwen2.5-7B-Instruct__253ebf694f1d/` |
| FLARE-direct (θ 0.8, β 0.4) | ASQA dev | 200 | 2.28 final claims; retrieval on 99.1% of look-ahead sentences | `results/dev/asqa/flare__Qwen2.5-7B-Instruct__180338fd77ba/` |
| Triage-and-Repair pipeline, τ = 0.5 | bio dev | 150 | 18.8 claims/response; 9 zero-claim responses; keep 2,454, revise 162, re-retrieve 96, low 102; 157 of 162 revisions DELETE, 2 supported; 27 of 96 re-retrievals supported | `results/dev/bio/longform__Qwen2.5-7B-Instruct__39454917f215/` |
| CoVe closed-book | bio dev | 150 | 5.33 final claims; 9 empty | `results/dev/bio/cove-closed_book__Qwen2.5-7B-Instruct__0eeb0eb0edd8/` |
| CoVe retrieval-verified | bio dev | 150 | 5.0 final claims; 17 empty | `results/dev/bio/cove-rag_verify__Qwen2.5-7B-Instruct__8addd1f8f25e/` |
| RARR-style | bio dev | 150 | 13.9 final claims; 170 edits | `results/dev/bio/rarr__Qwen2.5-7B-Instruct__8c772fdf2558/` |
| FLARE-direct | bio dev | 150 | 10.1 final claims; retrieval on 99.5% of sentences | `results/dev/bio/flare__Qwen2.5-7B-Instruct__e7c9cf098b08/` |

Log: `logs/phase4_longform_dev_v2.log`. Each is configuration #1 of its method (v2 does not count, D3).

### Short-form dev runs (G1, BM25 top-5, triage τ = 0.5) — match accuracy; exploratory

| Dataset | n | Standard RAG | A, no abstention | Repair fixed / broke | AURC CF / C / A | Coverage / accuracy-when-answering at target 0.90: CF; C; A | Output dir |
|---|---|---|---|---|---|---|---|
| PopQA | 500 | 0.312 | 0.308 | 3 / 5 | 0.543 / 0.471 / 0.555 | 0.21/0.52; 0.26/0.61; 0.20/0.49 | `results/dev/popqa/shortform__Qwen2.5-7B-Instruct__3e4402886b18/` |
| NQ | 500 | 0.318 | 0.310 | 3 / 7 | 0.555 / 0.561 / 0.558 | 0.21/0.51; 0.20/0.50; 0.21/0.51 | `results/dev/nq/shortform__Qwen2.5-7B-Instruct__930ec6943beb/` |
| TriviaQA | 500 | 0.652 | 0.642 | 10 / 15 | 0.171 / 0.172 / 0.172 | 0.61/0.83; 0.63/0.84; 0.61/0.84 | `results/dev/triviaqa/shortform__Qwen2.5-7B-Instruct__47c40550c092/` |
| HotpotQA | 500 | 0.444 | 0.390 | 8 / 35 | 0.432 / 0.382 / 0.453 | 0.28/0.66; 0.30/0.67; 0.28/0.65 | `results/dev/hotpotqa/shortform__Qwen2.5-7B-Instruct__4f88f27e3c75/` |
| 2WikiMultihopQA | 500 | 0.386 | 0.374 | 6 / 12 | 0.613 / 0.571 / 0.638 | 0.14/0.27; 0.21/0.53; 0.13/0.25 | `results/dev/2wiki/shortform__Qwen2.5-7B-Instruct__5c3f22c44c2d/` |

### Judged long-form results, ASQA dev (G1; judge = Phi-4 FP8, NOT YET VALIDATED against human labels) — exploratory

Judge: `RedHatAI/phi-4-FP8-dynamic` @ 8f62f4b8, prompt `judge_v1`, served with max-model-len 4096, GPU util 0.92
(KV cache 9,792 tokens). Evidence: ALCE oracle passages + reference long answers. Mean ± std over 100 seeded
evaluation halves (cal fraction 0.5). Source: `results/dev/asqa/comparison__v2__phi-4-FP8-dynamic.json`
(script `scripts/compare_longform.py`), `results/dev/asqa/longform__Qwen2.5-7B-Instruct__33f74812f8a7/analysis__phi-4-FP8-dynamic.json`
(script `scripts/analyze_longform.py`), log `logs/phase4_judge_asqa.log`.

| Method | Supported claims / answer | Claims kept / answer | Non-empty rate | Claim precision (micro) | P(all kept claims supported) |
|---|---|---|---|---|---|
| Standard RAG | 1.436 ± 0.139 | 2.219 ± 0.190 | 0.675 ± 0.034 | 0.648 ± 0.043 | 0.693 ± 0.033 |
| CoVe, closed-book | 1.067 ± 0.105 | 2.280 ± 0.188 | 0.730 ± 0.032 | 0.469 ± 0.034 | 0.528 ± 0.033 |
| CoVe, retrieval-verified | 1.085 ± 0.106 | 2.086 ± 0.209 | 0.629 ± 0.033 | 0.522 ± 0.037 | 0.617 ± 0.033 |
| RARR-style | 2.016 ± 0.184 | 4.530 ± 0.294 | 0.863 ± 0.023 | 0.445 ± 0.028 | 0.251 ± 0.032 |
| FLARE-direct | 1.436 ± 0.127 | 2.281 ± 0.221 | 0.687 ± 0.038 | 0.632 ± 0.038 | 0.667 ± 0.037 |
| CF @ 0.90 | 0.454 ± 0.116 | 0.640 ± 0.190 | 0.299 ± 0.070 | 0.720 ± 0.072 | 0.894 ± 0.047 |
| C @ 0.90 | 0.468 ± 0.149 | 0.626 ± 0.232 | 0.268 ± 0.080 | 0.764 ± 0.068 | 0.896 ± 0.048 |
| A @ 0.90 | 0.459 ± 0.121 | 0.648 ± 0.197 | 0.302 ± 0.071 | 0.719 ± 0.068 | 0.893 ± 0.046 |

A vs CF at 0.90 (partition 0, paired bootstrap 10,000): supported claims / answer diff +0.02, 95% CI [0.00, 0.06],
p = 0.745; precision diff −0.011, one-sided 95% lower bound −0.050 < −0.02 margin → guard does not hold.
Exploratory (dev); not a test of H2.

### Judge validation 1: Phi-4 FP8 judge vs FActScore's released human labels (EXPERIMENT_PLAN.md §7)

Script `scripts/validate_judge_factscore.py`; output `results/phase4/judge_validation_factscore.json`; log
`logs/phase4_judge_bio.log`. Evidence = FActScore KB, BM25 top-5 within the topic page, FActScore prompt layout.
Facts labelled IR by the annotators are excluded. κ CI = bootstrap (2,000 resamples, seed 0).

| System | Facts | Cohen's κ [95% CI] | Raw agreement | FActScore human / judge | Error (abs.) | Human S → judge NS | Human NS → judge S |
|---|---|---|---|---|---|---|---|
| InstructGPT | 4,071 | 0.735 [0.713, 0.756] | 0.868 | 0.474 / 0.513 | 0.039 | 193 | 345 |
| ChatGPT | 4,886 | 0.623 [0.600, 0.648] | 0.835 | 0.623 / 0.679 | 0.055 | 276 | 528 |
| PerplexityAI | 5,568 | 0.470 [0.437, 0.504] | 0.861 | 0.840 / 0.811 | 0.029 | 489 | 283 |
| Pooled | 14,525 | 0.652 [0.638, 0.665] | 0.854 | — | — | — | — |

Plan rule (κ ≥ 0.60 on both validations → accept): validation 1 passes on the pooled set. Validation 2: see below
(κ = 0.785, accepted). Caveats: κ for PerplexityAI is low under strong class imbalance; the judge
is lenient (it over-estimates FActScore for InstructGPT and ChatGPT by 3.9 and 5.5 points).

### Judge validation 2: Phi-4 FP8 judge vs 150 human labels on our outputs (EXPERIMENT_PLAN.md §7) — 2026-11-20

Sample: `scripts/make_labelling_sample.py` (seed 20261113) from the ASQA **dev** run
`results/dev/asqa/longform__Qwen2.5-7B-Instruct__33f74812f8a7`, judge tag `phi-4-FP8-dynamic`; stratified by judge
label (76 judge-supported, 74 not). Sources: 75 original claims (what standard RAG and CF filter) and 75 claims of
A's final candidate set; the "repaired text" pool was empty in that run (no repair produced new claim text), so its
quota moved to the other two (deviation from "50 each from A, CF and RAG"). Labels: one human labeller, Shivam Sengar (first
author), blind to source and judge label (did not open the key file; confirmed by the user 2026-11-20) (`labelling/human_labels.csv`, filled with `docs/LABELLING_GUIDE.md`; received 2026-11-20,
sha256 cf00c574165096db…). Script `scripts/score_labelling.py labelling/human_labels.csv` at commit df95f34 (+ the
labels file); output `labelling/agreement.json`. κ CI = bootstrap, 10,000 resamples, seed 0.

| Items | Human labels S / NS / U | Used (U excluded) | Cohen's κ [95% CI] | Raw agreement | Verdict (plan rule) |
|---|---|---|---|---|---|
| 150 | 71 / 78 / 1 | 149 | 0.785 [0.678, 0.879] | 0.893 | ACCEPT (κ ≥ 0.60) |

| Source | n | Agreement | Human supported rate | Judge supported rate |
|---|---|---|---|---|
| Original claims (RAG / CF) | 75 | 0.867 | 0.507 | 0.507 |
| A's final candidate claims | 74 | 0.919 | 0.446 | 0.500 |

Caveats: the sample is balanced by judge label, so κ is not on the natural label distribution; one labeller (no
human-human agreement); ASQA only (biographies are covered by validation 1). On A's claims the judge's supported rate
is 5.4 points above the human rate (lenient, as in validation 1).

### Failure analysis — human categorisation (EXPERIMENT_PLAN.md §8), test, G1, target 0.90, partition 0 — 2026-11-20

Sample: `scripts/failure_sample.py` (seed 0): 96 errors of A (50 correct-but-dropped, 46 wrong-but-kept; 21 bio +
25 ASQA kept). Categorised by Shivam Sengar (first author) in `labelling/failure_categories.csv` (built by
`scripts/make_failure_label_csv.py`, WITHOUT the assistant's pre-annotation; received 2026-11-20, sha256
f5daf6de41c3f7ea…, a note on every row). Script `scripts/score_failure_categories.py` at commit 57beab6 (+ the file);
output `results/test/failure_analysis_human.json`. These HUMAN counts replace the assistant pre-annotation in the paper.

| Cell | JE | RM | VFP | VFN | DE | UNC |
|---|---|---|---|---|---|---|
| bio, removed but supported (25) | 0 | 2 | 0 | 22 | 0 | 1 |
| ASQA, removed but supported (25) | 1 | 0 | 0 | 22 | 2 | 0 |
| bio, kept but unsupported (21) | 15 | 6 | 0 | 0 | 0 | 0 |
| ASQA, kept but unsupported (25) | 13 | 6 | 2 | 0 | 4 | 0 |

Agreement with the assistant pre-annotation: 66/96 = 0.688, Cohen's κ = 0.559 (main differences: the assistant's 11
UNC were mostly VFN (7) or JE (4) for the human; 4 assistant-JE kept claims are DE for the human). Kept RM (12), from
the annotator's notes: 9 different entity/work with a similar name (cases 003, 012, 016–019, 049, 050, 065), 3
misleading/outdated passage (052, 059, 063); re-retrieval raised 012 and 063 above the threshold.

### Pilot quantity R (EXPERIMENT_PLAN.md §1), dev, G1

| Dev set | Dropped by CF @ 0.90 | R (preregistered) | R_net (wrong → right) | R_kept (also passes filter) | Source |
|---|---|---|---|---|---|
| HotpotQA (short-form) | 358 answers | 0.140 | 0.022 | 0.006 | `results/dev/hotpotqa/shortform__Qwen2.5-7B-Instruct__4f88f27e3c75/pilot.json` (`scripts/pilot_shortform.py`) |
| ASQA (long-form; NOT part of the preregistered pilot) | 319 claims | 0.047 | 0.000 | 0.009 | `results/dev/asqa/longform__…33f74812f8a7/analysis__phi-4-FP8-dynamic.json` |
| Biographies (long-form) | 2,708 claims | 0.029 | 0.000 | 0.0004 | `results/dev/bio/longform__Qwen2.5-7B-Instruct__39454917f215/analysis__phi-4-FP8-dynamic.json` |
| **Preregistered pilot: biographies + HotpotQA pooled** | 3,066 | **0.042** (= (78 + 50) / (2,708 + 358)) | 0.007 | 0.001 | computed from the two rows above |

**Pilot decision (rule fixed in EXPERIMENT_PLAN.md §1 before any result): R = 0.042 < 0.05 → proceed with all
planned runs, frame the paper as a controlled comparative study, and state plainly that repair did not work in this
setup.** Decided 2026-11-16 on dev data. Caveat: the judge's second validation (user labels) is still pending.

### Judged long-form results, biography dev (G1; same judge; FActScore KB evidence) — exploratory

Source: `results/dev/bio/comparison__v2__phi-4-FP8-dynamic.json`; `results/dev/bio/longform__…39454917f215/analysis__phi-4-FP8-dynamic.json`;
log `logs/phase4_judge_bio.log`. Mean ± std over 100 evaluation halves (75 responses each).

| Method | Supported claims / answer | Claims kept / answer | Non-empty rate | Claim precision (micro) | P(all kept claims supported) |
|---|---|---|---|---|---|
| Standard RAG | 15.503 ± 0.642 | 18.806 ± 0.603 | 0.941 ± 0.019 | 0.824 ± 0.019 | 0.213 ± 0.031 |
| CoVe, closed-book | 2.174 ± 0.238 | 5.398 ± 0.369 | 0.634 ± 0.036 | 0.402 ± 0.030 | 0.380 ± 0.036 |
| CoVe, retrieval-verified | 2.501 ± 0.274 | 5.065 ± 0.371 | 0.666 ± 0.036 | 0.493 ± 0.033 | 0.375 ± 0.037 |
| RARR-style | 5.909 ± 0.527 | 13.950 ± 0.737 | 0.753 ± 0.034 | 0.423 ± 0.027 | 0.299 ± 0.035 |
| FLARE-direct | 6.618 ± 0.445 | 10.229 ± 0.556 | 0.718 ± 0.033 | 0.647 ± 0.024 | 0.354 ± 0.036 |
| CF @ 0.90 | 0.641 ± 0.276 | 0.768 ± 0.350 | 0.409 ± 0.099 | 0.843 ± 0.049 | 0.904 ± 0.049 |
| C @ 0.90 | 1.403 ± 0.385 | 1.523 ± 0.446 | 0.422 ± 0.074 | 0.926 ± 0.030 | 0.909 ± 0.045 |
| A @ 0.90 | 0.643 ± 0.278 | 0.771 ± 0.352 | 0.410 ± 0.099 | 0.844 ± 0.048 | 0.904 ± 0.049 |
| SelfCheckGPT-NLI @ 0.90 | 1.858 ± 0.329 | 1.971 ± 0.371 | 0.579 ± 0.070 | 0.945 ± 0.021 | 0.899 ± 0.050 |

A vs CF at 0.90 (partition 0): identical (diff 0.0). Other targets (0.80–0.95) are in the analysis files.
ASQA re-analysis with SelfCheckGPT: at 0.90 SelfCheckGPT keeps 0.349 ± 0.103 supported claims / answer (CF 0.454,
C 0.468, A 0.459), precision 0.706 — on ASQA it is the weakest filter. Source `results/dev/asqa/comparison__v2__phi-4-FP8-dynamic.json`.

The preregistered pilot pools biographies + HotpotQA; see the decision above.
R_net and R_kept are stricter variants added on 2026-11-16 after seeing R; they are reported, not used for the rule.

Scripts: `scripts/run_shortform.py`, `scripts/analyze_shortform.py` (100 seeded calibration/evaluation partitions,
cal fraction 0.5); log `logs/phase4_shortform_dev.log`. Each is configuration #1 of A / C / CF on short-form.
Coverage/accuracy cells are means over partitions. No significance tests were run (dev data, exploratory).

## TEST runs (Phase 5) — raw tables, G1 = Qwen2.5-7B-Instruct

### Short-form closed-book and standard RAG (BM25 top-5), n = 1,000 per set (fixed seeded test subset), greedy

Script `scripts/run_qa.py` via `scripts/phase5_all.sh`; commit of results 1df2639; log `logs/phase5_all.log`.
Output `results/test/<set>/{closed_book,rag}__Qwen2.5-7B-Instruct__<hash>/summary.json`.

| Set | closed-book match acc | closed-book EM | standard RAG match acc | RAG EM | RAG F1 | retrieval recall@5 | closed-book dir hash | RAG dir hash |
|---|---|---|---|---|---|---|---|---|
| PopQA | 0.176 | 0.145 | 0.316 | 0.254 | 0.298 | 0.416 | 2af834d34508 | f1efe9be936b |
| NQ | 0.237 | 0.155 | 0.341 | 0.250 | 0.357 | 0.476 | 101a7e41b367 | 36a8454ad311 |
| TriviaQA | 0.538 | 0.467 | 0.678 | 0.575 | 0.660 | 0.753 | c67c61ad213f | bf12e44945fc |
| HotpotQA | 0.210 | 0.180 | 0.363 | 0.321 | 0.428 | 0.482 | 639c2f1032f4 | 82dbc57dc7a6 |
| 2Wiki | 0.239 | 0.230 | 0.266 | 0.255 | 0.308 | 0.335 | 77f5e1f09bf5 | 3a1a8b407bd3 |

### Same, G2 = Mistral-7B-Instruct-v0.3 (table from `scripts/qa_table.py --model-tag Mistral-7B-Instruct-v0.3`)

| Set | closed-book match acc | closed-book EM | standard RAG match acc | RAG EM | RAG F1 | retrieval recall@5 | closed-book dir hash | RAG dir hash |
|---|---|---|---|---|---|---|---|---|
| PopQA | 0.250 | 0.151 | 0.342 | 0.196 | 0.276 | 0.416 | 3b9c63e741f5 | 440938681dbe |
| NQ | 0.338 | 0.094 | 0.366 | 0.140 | 0.280 | 0.476 | af2f1f006678 | 0dfe9b41d090 |
| TriviaQA | 0.687 | 0.422 | 0.711 | 0.490 | 0.620 | 0.753 | d39483165503 | d34fcdef21df |
| HotpotQA | 0.262 | 0.066 | 0.399 | 0.157 | 0.336 | 0.482 | 5fc89bce1692 | d0ccf00dfa57 |
| 2Wiki | 0.282 | 0.011 | 0.347 | 0.039 | 0.210 | 0.335 | 0959c66f5c23 | bc9d935066e3 |

### Same, G3 = Falcon3-7B-Instruct (`scripts/qa_table.py --model-tag Falcon3-7B-Instruct`)

| Set | closed-book match acc | closed-book EM | standard RAG match acc | RAG EM | RAG F1 | retrieval recall@5 | closed-book dir hash | RAG dir hash |
|---|---|---|---|---|---|---|---|---|
| PopQA | 0.157 | 0.144 | 0.335 | 0.260 | 0.315 | 0.416 | ac568026d50b | a6c188579cce |
| NQ | 0.217 | 0.155 | 0.376 | 0.268 | 0.382 | 0.476 | b44411acd8a6 | dbbc2a5a107c |
| TriviaQA | 0.511 | 0.474 | 0.701 | 0.617 | 0.690 | 0.753 | 0c4dfdaf460c | c77ed24752c5 |
| HotpotQA | 0.193 | 0.181 | 0.370 | 0.313 | 0.435 | 0.482 | 0986bb062b38 | 028da3969250 |
| 2Wiki | 0.251 | 0.244 | 0.314 | 0.298 | 0.354 | 0.335 | 499b23b48d38 | d1248c287000 |

Observation (not a test): Mistral's EM is far below its match accuracy (e.g. 2Wiki 0.039 vs 0.347) because it answers in
full sentences; match accuracy (answer string contained) is the preregistered primary short-form metric, but
cross-generator EM comparisons are therefore not meaningful and the paper must say so.

### Long-form TEST, verification pipelines, baselines, confirmatory tests; short-form selective metrics (all generators)

All tables are generated (no hand transcription) by `scripts/ledger_tables.py` into
**`results/test/tables_phase5.md`** (T1 G1 all methods; T2 all generators × targets; T3 A vs CF + pilot R; T4 short-form).
Inputs: `results/test/*/longform__*/analysis__phi-4-FP8-dynamic.json` (`scripts/analyze_longform.py`),
`results/test/{asqa,bio}/comparison__0.95__v2__phi-4-FP8-dynamic.json` (`scripts/compare_longform.py`),
`results/test/confirmatory.json` (`scripts/confirmatory.py`), `results/test/*/shortform__*/analysis.json`
(`scripts/analyze_shortform.py`); all run by `scripts/phase5_analyze.sh` (log `logs/phase5_analyze.log`) at commit
4b94969 (+ uncommitted docs only). Judge = Phi-4 FP8 (validation 1 only; validation 2 pending human labels).
100 seeded calibration/evaluation partitions (cal 0.5); paired bootstrap 10,000, seed 0, partition 0.

**Confirmatory result (preregistered, G1, α = 0.05, Holm over H1/H2): H1 (biographies) NOT supported — Δ supported
claims/answer A − CF = +0.034, 95% CI [0.000, 0.080], p = 0.100; H2 (ASQA) NOT supported — Δ = −0.011,
CI [−0.027, 0.005], p = 0.267.** Non-inferiority guard holds in both. Secondary (uncorrected) G2/G3: Mistral +0.051
(bio, p = 0.0002) and +0.029 (ASQA, p < 1e-4) with guard holding; Falcon3 +0.040 (bio, p = 0.004) but guard FAILS
(precision −0.022, lower bound −0.051) and +0.011 (ASQA, p = 0.55, guard fails). Test pilot R (G1): bio 0.040, ASQA 0.059.

## Phase 6 analyses on TEST — ALL EXPLORATORY (plan §10: unadjusted, labelled exploratory)

| Analysis | Script | Output | Key numbers (see output for all cells) |
|---|---|---|---|
| Pairwise long-form comparisons at each target (C vs CF, SelfCheckGPT vs CF, C vs SelfCheckGPT), all generators | `scripts/exploratory_tests.py` | `results/test/exploratory_tests.json`; tables T5 in `results/test/tables_phase5.md`; log `logs/phase6_exploratory.log` | @0.90 C − CF supported/answer: bio G1 +1.82 [1.30, 2.39], Mistral +1.27 [0.96, 1.61], Falcon3 −0.21 [−0.47, 0.06] (n.s.); ASQA G1 +0.16 [0.06, 0.27], Mistral +0.13 [0.04, 0.22], Falcon3 +0.14 [0.05, 0.23] |
| Short-form McNemar (A vs RAG accuracy) and AURC(C) − AURC(CF) | same | same, table T6 | A never more accurate than RAG; significantly less accurate in 8/15 cells. C lower AURC significantly in 4/15, higher in 1/15 |
| Claim-level AUROC vs judge label (s, C score, c, 1 − con, SelfCheckGPT) | `scripts/claim_auroc.py` | `results/test/claim_auroc.json` | all 0.54–0.69; C > s in 5/6 cells; Falcon3-bio c AUROC 0.537 (≈ chance) |
| Ablations §8.2 / §8.3 from stored repairs | `scripts/ablation_table.py` | `results/test/ablations.json`, `results/test/tables_ablations.md` | revision-only ≡ CF (Δ = 0.000 on both sets); A's whole effect comes from re-retrieval |
| Failure analysis §8: 96 errors of A (G1, target 0.90, partition 0; bio has only 21 wrong-but-kept) | `scripts/failure_sample.py`, `scripts/failure_summary.py` | `labelling/failure_sample.csv`, `labelling/failure_preannotation_assistant.json`, `results/test/failure_analysis_preannotation.json` | **ASSISTANT PRE-ANNOTATION, NOT HUMAN — must be verified before reporting.** wrong-but-kept: judge-evidence scope 30/46, retrieval wrong entity 7, verifier FP 5; correct-but-dropped: supported but s below q̂ 37/50 |
| §8 ablations + sensitivity, G1 long-form (one-axis, τ_s ∈ {0.3,0.4,0.6,0.7}, k ∈ {3,10}, sentences, verifier NLI / AlignScore / generator) and size (3B, 14B-AWQ) | `scripts/phase6_runs.sh` → `scripts/ablation_table.py` | `results/test/tables_ablations.md`, `results/test/ablations.json`; runs logged in `logs/phase6_runs.log` (judged 2026-11-18 05:22–06:43Z) | A ≈ CF everywhere (largest: 14B ASQA +0.067 [0.032, 0.107]; one-axis bio +0.040 [0.006, 0.091], p = 0.036). C > CF on bio under every swap / k / 14B; on ASQA not with AlignScore, 14B, sentences; generator-verifier ASQA C < CF (−0.139). Alternative verifiers keep more than MiniCheck under CF (bio 1.02–1.13 vs 0.74) |
| Generator size, short-form (3B, 14B-AWQ): QA tables + selective metrics | `scripts/analyze_sizes.sh` (qa_table + analyze_shortform) | `logs/phase6_sizes.log`; `results/test/*/shortform__Qwen2.5-{3B-Instruct,14B-Instruct-AWQ}__*/analysis.json` | RAG > closed-book at all sizes; A accuracy ≤ RAG in 10/10 cells (14B HotpotQA 0.348 vs 0.405); C AURC < CF AURC in 9/10 (descriptive) |
| Dense-retriever sensitivity (bge-small, exact search over 21,015,324 passages), G1 long-form @0.90 | `scripts/phase6_runs.sh` → `scripts/ablation_table.py` | `results/test/tables_ablations.md` rows "retriever=dense" (runs `longform__Qwen2.5-7B-Instruct__dbd130b24564` bio, `__cf7f9adc1fca` ASQA; index meta `~/halluc_data/index/dense_bge_small/build_meta.json`); `logs/phase6_runs.log` (PHASE6_RUNS_END 2026-11-18 12:30:33Z) | bio: CF 0.888, C 2.158, A 0.918; A−CF +0.029 [0.006, 0.057] p 0.013; C−CF +1.531. ASQA: CF 0.685, C 0.782, A 0.689; A−CF −0.035 [−0.064, −0.005] p 0.031; C−CF −0.099 [−0.219, 0.024] p 0.118 (exploratory) |
| Per-question cost by method (RQ2), long-form G1 — logged calls/tokens only; no wall-clock (stage seconds include cache hits) | `scripts/cost_table.py` | `results/test/cost_table.md` | bio: RAG 1.0 call / 875 prompt tok; CF 2.0 / 1,548; C 19.8 / 2,740; A 20.8 / 3,009; SelfCheckGPT 7.0 / 5,925 (+ sample completions not logged); CoVe-RAG 18.5 / 9,041; RARR 11.7 / 6,050; FLARE 15.3 / 8,840 |
| Claims removed by CF @0.90 vs repairs attempted, TEST, G1 (paper §6) | `scripts/analyze_longform.py` field `pilot` (q̂ on the full test run) | `results/test/{bio,asqa}/longform__Qwen2.5-7B-Instruct__*/analysis__phi-4-FP8-dynamic.json` | bio: dropped 5,923, attempted 638 (10.8%), repaired judge-supported 237 (R 0.040); ASQA: dropped 1,208, attempted 237 (19.6%), 71 (R 0.059) |
| Revision outcomes on TEST, G1 (paper §5.2) | `scripts/run_longform.py` summary field `repair` | `results/test/bio/longform__Qwen2.5-7B-Instruct__5aa2ad1482e5/summary.json`, `results/test/asqa/longform__Qwen2.5-7B-Instruct__ceeba04af770/summary.json` | bio: revise attempted 361, DELETE 348, now supported 5; re-retrieve attempted 277, now supported 85. ASQA: revise 123, DELETE 123, 0; re-retrieve 114, 58 |
| Claim-level calibration (ECE, 10 bins) of each score vs judge label | `scripts/claim_auroc.py` | `results/test/claim_auroc.json` | ECE 0.12–0.46; c worst (bio G1 0.459); SelfCheckGPT lowest (bio 0.116, ASQA 0.170) |
| Budgeted repair B vs random subset of equal size (§5.3), G1, @0.90 | `scripts/budget_random.py` | `results/test/budget_random.json` | no difference: bio B=1/2/4 risk 0.745/0.750/0.730 vs random 0.740/0.740/0.754; ASQA all 0.536–0.540 |
| CF / C threshold at target 0.90 on TEST, G1 (median and range over partitions 0–99, cal 0.5) and share of judge-supported / unsupported claims scoring above the median threshold (paper §6, Fig. score_distributions) | `scripts/figures_extra.py` (uses `hallu.analysis.evaluate_once`) | `results/test/figure_numbers.json` | bio s: 0.9794 [0.9764, 0.9818], above 5.1% sup / 3.9% unsup; ASQA s: 0.9739 [0.9675, 0.9760], 32.1% / 18.5%; C score: bio 1 − 1.3e-7 (13.5% / 5.5%), ASQA 1 − 9.4e-5 (40.7% / 21.5%). Replaces the dev-set "0.98" previously cited in §6 (same value to two decimals) |
| Paper figures added 2026-11-19 (no new runs): retention for Mistral-7B / Falcon3-7B, precision vs supported claims (Qwen), short-form risk–coverage (Qwen), claim AUROC with CI, score distributions | `scripts/figures_extra.py`, `scripts/figures.py` | `results/figures/test/{retention_other_generators,precision_vs_content,shortform_risk_coverage__Qwen2.5-7B-Instruct,claim_auroc,score_distributions}.pdf`; inputs: default-run `analysis__phi-4-FP8-dynamic.json`, `results/test/{bio,asqa}/comparison__0.95__v2__phi-4-FP8-dynamic.json` (Qwen runs; file name from a naming bug fixed in `compare_longform.py`), `results/test/claim_auroc.json`, items + judge files | — |
| NOT DONE (deviations to report): RAGTruth verifier validation (needs a dataset download, disallowed mid-project by the user 2026-11-17); ALCE citation precision/recall (needs the 45 GB TRUE NLI model; final answers carry no citations); single-stream latency (not measured) | — | — | — |
| Generator size, observation (not a test): Qwen2.5-3B cannot run the long-form pipeline — its own decomposition returns "No claims." for most answers (bio 67 claims for 350 answers; ASQA 513 for 748; 14B-AWQ: 5,330 / 1,936; G1: 6,230 / 1,692) | `scripts/count_refusals.py` (LLM-cache diagnostic) | `logs/phase6_runs.log` (STOP_AFTER decompose lines) | 3B cache: 862 "No claims." vs 232 bulleted decompositions; ~203 of its long answers decline. Long-form size results for 3B are reported as this failure, not as filter comparisons |

## Residual serving non-determinism (EXPERIMENT_PLAN.md §10), TEST, G1

PopQA test standard RAG, cached run vs 3 cache-free repeats (`scripts/run_qa.py --cache-tag repK`):
accuracy 0.316 in all four (std 0.000); 999 / 1,000 predictions identical across all four runs.
Source `results/test/determinism_popqa_rag.json` (`scripts/determinism.py`). Greedy-decoded results are therefore
reported from single runs; the one differing item does not change any aggregate.

## Why repair fails — diagnosis on DEV data (G1), 2026-11-16

Script `scripts/why_repair_fails.py`; outputs `results/dev/{bio,asqa}/longform__…/why_repair_fails__phi-4-FP8-dynamic.json`.
AUROC = how well a claim score separates judge-supported from unsupported claims (`hallu.metrics.auroc`, tested).

| Quantity | Biographies (2,814 claims) | ASQA (443 claims) |
|---|---|---|
| Judge-supported rate of all claims | 0.825 | 0.652 |
| AUROC: MiniCheck support s (CF score) | 0.615 | 0.588 |
| AUROC: two-signal score (C) | 0.628 | 0.592 |
| AUROC: closed-book confidence c alone | 0.583 | 0.570 |
| AUROC: 1 − SelfCheckGPT contradiction | 0.687 | 0.616 |
| CF threshold at target 0.90 (full dev) | 0.980 | 0.975 |
| Claims above that threshold | 106 | 124 |
| Median support s of all claims | 0.954 | 0.961 |
| Revise branch: n / judge says original supported / DELETE / revised text supported | 162 / 82 / 157 / 2 | 27 / 13 / 27 / 0 |
| Re-retrieve: n / originally supported / support rose / crossed CF threshold | 96 / 76 / 42 / 1 | 26 / 15 / 16 / 4 |
| Re-retrieve: median s before → after | 0.070 → 0.217 | 0.140 → 0.264 |
| "Low" bucket: n / judge-supported | 102 / 69 | 17 / 7 |

Reading: (1) every available score ranks claims only weakly (AUROC 0.57–0.69 against the judge); (2) support scores
pile up near 1, so a 0.90 target puts the threshold at the extreme tail (≈ 0.98) where repair cannot lift claims;
(3) triage routes on noisy signals — half the "contradicted" claims are judge-supported and revision deletes them.
SelfCheckGPT ranks best, which matches it retaining the most supported claims at matched targets. Caveat: AUROCs are
against the (lenient) judge, not humans.

## Generator / judge fit tests (EXPERIMENT_PLAN.md §2; infrastructure, not paper results)

Script `scripts/fit_test.sh` (vLLM 0.33.0 in Docker, BF16 unless stated, max-model-len 8192, GPU util 0.90; rule:
loads and ≥ 16,000 KV-cache tokens). Outputs `results/phase4/fit/*.json`.

| Model @ revision | Load GiB | KV tokens | Rule | Single-stream tok/s | 32-concurrent tok/s |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct @ a09a3545 (G1, Phase 0) | 14.29 | 37,920 | pass | 22.4 | 665.1 |
| mistralai/Mistral-7B-Instruct-v0.3 @ c170c708 (G2) | 13.51 | 30,352 | pass | 22.9 | 664.2 |
| ibm-granite/granite-4.2-8b @ a7042144 | 16.42 | (0.12 GiB) | FAIL | — | — |
| allenai/Olmo-3-7B-Instruct @ 6e5971d9 | 13.59 | (~6,200 max) | FAIL | — | — |
| tiiuae/Falcon3-7B-Instruct @ 1e57a0ec (G3) | 13.93 | 25,344 | pass | 23.0 | 660.9 |
| RedHatAI/phi-4-FP8-dynamic @ 8f62f4b8 (judge; 4096 ctx, util 0.92) | 14.74 | 9,792 | n/a (judge) | — | — |
| selfrag/selfrag_llama2_7b @ 19026138 (float16, 4096 ctx) | 12.56 | 9,472 | n/a (baseline) | — | — |

## Compute estimate (EXPERIMENT_PLAN.md §12), 2026-11-16

`scripts/estimate_compute.py`, from measured dev costs scaled to test sizes: Tier 1 (G1, every method in the table
of long-form and short-form dev runs) ≈ 3.0 GPU-h of generation + 0.6 h of verifiers on GPU; Tier 2 (G2, G3; the
two pipeline runs) ≈ 2.5 h. Not included: judging (~3–5 judge calls/s measured), SelfCheckGPT samples, Self-RAG,
ablations, tuning. Far below the plan's 10-GPU-day trigger, so no reduction is proposed. Wall-clock is dominated by
CPU verification when the GPU serves a generator and by unreliable downloads.

## Sanity reproduction (EXPERIMENT_PLAN.md §11) — original setup, reported number vs ours

| Item | Ours | Reported in paper | Gap | Script | Output |
|---|---|---|---|---|---|
| Self-RAG 7B (`selfrag/selfrag_llama2_7b` @ 19026138), PopQA long-tail (1,399), official `match`, adaptive retrieval, threshold 0.2, 10 Contriever passages, w_rel = w_sup = w_use = 1, use_seqscore | **0.5497** | 0.549 (Asai et al., Table 2) | +0.0007 | `scripts/selfrag_repro.py` (port of the official `run_short_form.py`), code commit 726c8bf | `results/phase4/selfrag_repro/summary.json`, `predictions.jsonl` |

Data: third-party copy of the official `popqa_longtail_w_gs.jsonl` (two independent Hub uploads, byte-identical,
SHA-256 3b9ae00f…efc3, 1,399 rows = paper). Served by vLLM 0.33.0 in float16, max-model-len 4096, greedy.
Our normalised match on the same predictions: 0.5583. Retrieval frequency 1.000 — a consequence of the official
retrieve-or-not rule dividing raw log-probabilities, which the port keeps on purpose. Cost: 15,389 requests,
2.50 M prompt + 0.32 M completion tokens, 804 s. Log `logs/phase4_gpu_queue.log`.

### Self-RAG 7B as a short-form baseline on our dev sets (own row group: fine-tuned Llama-2 generator, NOT like-for-like)

Official checkpoint, ported decoding (same settings as the reproduction), passages = OUR BM25 top-10 (Self-RAG's
ndocs = 10; our pipelines use top-5). Script `scripts/selfrag_repro.py --dataset D --split dev`; outputs
`results/dev/<D>/selfrag__selfrag_llama2_7b/summary.json`; logs `logs/phase4_gpu_queue.log`, `logs/phase4_gpu_queue_b.log`.

| Dataset (dev, n = 500) | Normalised match (ours) | Official raw `match` | Retrieval frequency |
|---|---|---|---|
| PopQA | 0.356 | 0.350 | 1.00 |
| NQ | 0.318 | 0.274 | 1.00 |
| TriviaQA | 0.704 | 0.558 | 1.00 |
| HotpotQA | 0.412 | 0.378 | 1.00 |
| 2WikiMultihopQA | 0.312 | 0.190 | 1.00 |

NQ and TriviaQA first failed ("Event loop is closed", client reused across event loops) and were re-run after the fix
(commit 76edcca); the failed attempts produced no output.

### Self-RAG 7B, TEST (n = 1,000 per set; same settings; own row group, NOT like-for-like)

Script `scripts/selfrag_repro.py --dataset D --split test` via `scripts/phase5_all.sh` (2026-11-17 19:11–20:21Z,
vLLM fp16, max-model-len 4096); outputs `results/test/<D>/selfrag__selfrag_llama2_7b/summary.json`; log
`logs/phase5_all.log`. "Normalised match (ours)" is the same match function as every other short-form row and is the
column to compare; the official raw `match` is shown for transparency (it differs most on TriviaQA and 2Wiki).

| Dataset (test) | Normalised match (ours) | Official raw `match` | Retrieval frequency |
|---|---|---|---|
| PopQA | 0.389 | 0.379 | 1.00 |
| NQ | 0.320 | 0.288 | 1.00 |
| TriviaQA | 0.710 | 0.556 | 1.00 |
| HotpotQA | 0.362 | 0.317 | 1.00 |
| 2WikiMultihopQA | 0.344 | 0.217 | 1.00 |

## Run log (all runs, chronological, including failures)

| Date | Phase | Run | Outcome | Log |
|---|---|---|---|---|
| 2026-11-12 | 0 | Build env (`scripts/setup_env.sh`) | OK | `logs/phase0_setup_env.log` |
| 2026-11-12 | 0 | torch GPU check | OK | `logs/phase0_gpu_and_download.log` |
| 2026-11-12 | 0 | Model download via Xet | FAILED (CAS client error after ~11 GB) | `logs/phase0_gpu_and_download.log` |
| 2026-11-12 | 0 | Model download via plain HTTPS | OK (15.24 GB) | `logs/download_Qwen_Qwen2.5-7B-Instruct_20261112_100020.log` |
| 2026-11-12 | 0 | vLLM serve, default settings | FAILED (no C compiler in WSL) | `logs/vllm_serve_20261112_101757.log` |
| 2026-11-12 | 0 | vLLM serve, `-cc.mode=none` | FAILED (no C compiler in WSL) | `logs/vllm_serve_20261112_101912.log` |
| 2026-11-12 | 0 | Docker pull `vllm/vllm-openai:v0.33.0`, first tmux session | KILLED (WSL restarted) | `logs/docker_pull_vllm_20261112_102447.log` |
| 2026-11-12 | 0 | Docker pull, second session | OK on attempt 16 (15 failed: 1 short read, 14 registry timeouts) | `logs/docker_pull_vllm_20261112_104030.log` |
| 2026-11-12 | 0 | vLLM serve in Docker | OK | `logs/vllm_docker_20261112_114143.log` |
| 2026-11-12 | 0 | vLLM throughput benchmark | OK (216 requests, all finished at max_tokens) | `logs/phase0_bench.log` |
| 2026-11-13 | 4 | Corpus download `psgs_w100.tsv.gz` | OK (4.69 GB, sha256 c39b020c…) | `logs/dl_corpus.log` |
| 2026-11-13 | 4 | ALCE data download | OK (451 MB) | `logs/dl_alce.log` |
| 2026-11-13 | 4 | BM25 smoke build (200k passages), first attempt | FAILED (wrong input type for bm25s `index`) → fixed | session output |
| 2026-11-13 | 4 | BM25 full build (21,015,324 passages) | index OK; self-check FAILED on a bad test query ("xxyyzz" exists in Wikipedia) → query fixed, self-check OK | `logs/bg_bm25.log` |
| 2026-11-13 | 4 | Fixed dev/test splits, 5 QA datasets | OK, rebuilt identically on re-run | `logs/phase4_build_splits.log`, `data/splits/MANIFEST.json` |
| 2026-11-13 | 4 | Closed-book and RAG dev runs, G1 | OK (see dev table) | `logs/phase4_closed_book_dev.log`, `logs/phase4_rag_dev.log` |
| 2026-11-13 | 4 | FActScore `data.zip` | OK (2.2 MB) | `logs/dl_factscore.log`, `logs/phase4_factscore_data.log` |
| 2026-11-13 | 4 | FActScore `enwiki-20230401.db` (21.9 GB) | **FAILED at 9.2 GB**: Google Drive quota ("too many users"); partial file kept | `logs/dl_factscore.log` |
| 2026-11-13 | 4 | Verifier sanity check (MiniCheck + DeBERTa NLI, CPU) | OK after pinning official weights; NLI calls an off-topic false claim "contradiction" (known failure mode) | `logs/phase4_verifier_sanity.log` |
| 2026-11-13 | 4 | Unit tests | 41 passed | `logs/pytest_latest.log` |
| 2026-11-16 | 5 | Phase 5 master `scripts/phase5_all.sh`, G1 gen stage | Interrupted ~12:04Z at CoVe (WSL restart); resumed 2026-11-17 04:32Z via `scripts/start_phase5.sh` (cached; finished short-form runs skipped) | `logs/phase5_all.log` |
| 2026-11-17 | 5 | G1, G2, G3 gen stages | OK (GEN_STAGE_END for all three) | `logs/phase5_all.log` |
| 2026-11-17 | 5 | GPU score stage | **FAILED** 13:07Z: torch 2.14 native Triton op `bmm_outer_product` needs a C compiler (none in WSL). Fixed with `TORCH_DISABLE_NATIVE_JIT=1` in `scripts/wsl_env.sh` (stock torch kernels; torch/transformers versions unchanged); G1 ASQA scoring then OK (8,460 pairs) | `logs/phase5_all.log`, `scripts/debug_score.sh` |
