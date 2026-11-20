# Phase 2 — Candidate methods (novelty-checked)

_Written 2026-11-13. Nothing here has been implemented or run. Every expected benefit below is a hypothesis._

## 0. What the novelty check found (read this first)

Searches: 13 all-year arXiv queries (`refs/search_queries_phase2.txt`, 55 unique hits,
`logs/phase2_arxiv_search.log`), 12 abstracts read, 8 papers read in targeted sections
(`scripts/lit/read_phase2.sh`). Keyword search can miss papers; "not found" is not proof of absence.

Three findings changed the candidates I had in mind after Phase 1:

1. **Claim-level calibrated filtering for RAG is already a crowded line.** Conformal factuality
   (Mohri & Hashimoto, 2402.10978) decomposes an answer into claims and removes low-scoring ones under a
   calibrated threshold. It has been applied to RAG (Conformal-RAG, 2506.20978), stress-tested
   (Chen et al., 2603.16817), and extended to multi-hop and agentic RAG (Rehman et al., 2609.38222, 2610.34872).
   A "verify claims, then abstain with calibration" method would be incremental.
2. **But those papers document a specific, measurable failure: filtering destroys the answer.**
   - Rehman et al. (their single-hop reference experiment, HotpotQA, Llama 3.1 8B): at a 95% target, fully-supported responses rise from 61.67% to
     96.27%, while **claim retention falls to 11.10% and only 20.97% of responses stay non-empty**
     (60-question subsets; their stated limitation).
   - Chen et al.: "conformal filtering suffers from low usefulness at high factuality levels due to vacuous
     outputs", is not robust to distractors, and "this raises the need for new approaches". They also report that
     lightweight verifiers can beat larger LLM scorers at orders-of-magnitude fewer FLOPs.
   - In none of the four conformal-RAG papers did I find a step that *repairs* a low-scoring claim
     (re-retrieval or revision) before deciding to drop it. They only filter.
3. **A decision policy over "parametric vs contextual reliability" exists at answer level.**
   SABER (2605.18792) estimates whether the parametric answer and the context-conditioned answer are each
   correct and chooses trust-PK / trust-CK / trust-both / abstain. It is short-form conflict QA, uses probes on
   hidden states trained on ~69K labelled instances per backbone, and (authors' limitation) "assumes that the
   retrieved context has already been supplied" — no retrieval, no claim level, no revision.

Other neighbours checked: EAVer (2609.22223) learns a search-efficient *evaluation* policy by fine-tuning
Qwen3-8B (about 80% fewer searches) — cost-aware verification exists on the evaluator side, with training.
CoVe-RAG+ (2609.26229) is CoVe + RAG in an engineering/CAD setting with a fine-tuned generator. ArbGraph
(2604.18362) arbitrates conflicting evidence *before* generation. The medical atomic fact-checking paper
(2505.24830) could not be fetched in full (no HTML version); abstract only.

**Consequence.** The defensible opening is not "verify and abstain" (done) and not "detect and regenerate"
(SeaRAG). It is: *between detection and deletion, try to repair — and show, under one setup, whether that buys
back the informativeness that filtering throws away, and at what cost.*

---

## Candidate A — Triage-and-Repair before filtering  (recommended core)

**Idea.** Generate with standard RAG, decompose the answer into atomic claims, and score each claim on two axes:
(i) support by retrieved evidence (small NLI-style verifier) and (ii) the generator's own confidence in the claim
without context. A fixed policy maps the result to an action:

| Evidence support | Parametric confidence | Action |
|---|---|---|
| supported | any | keep, attach citation |
| not supported | high | targeted re-retrieval for that claim, re-verify; keep only if now supported |
| contradicted | any | revise the claim from the contradicting evidence, re-verify |
| not supported | low | drop |

Only after this repair pass is a calibrated threshold applied (the same split-conformal filter used by prior work),
and the answer is abstained on if nothing survives.

**Closest prior work and the difference.**
- Conformal-RAG / Chen et al. / Rehman et al.: filter only. A adds repair before the filter and is evaluated on
  *their* failure metric (claim retention and non-empty rate at a fixed factuality target).
- RARR (2210.08726): revises everything post hoc with no verification-driven selection, no abstention, no
  calibration; reported attribution gains were modest.
- SeaRAG (WWW 2026): triggers on the model's self-reported confidence, regenerates whole sentences, never checks
  the result against evidence, no abstention.
- FRANQ (2505.21072): supplies the faithful-vs-factual split for detection and names generation-time control as
  future work. A is one concrete form of that control.
- SABER: similar two-axis logic, but answer-level, short-form, trained probes, no retrieval or revision.
- VeriCite (2510.11394): NLI-verified statements for citation quality; no re-retrieval, no abstention.

**Honest novelty rating: moderate.** Every component exists. The contribution would be (1) the repair-then-filter
ordering with a two-axis policy at claim level, and (2) evidence on whether it moves the retention–factuality
frontier. If it does not, the paper has no method contribution (see risks).

**Falsifiable primary hypothesis.** At a fixed target factuality level, Triage-and-Repair retains a larger
fraction of claims and yields more non-empty answers than filter-only conformal factuality, with factual precision
not lower. To be fixed precisely (levels, datasets, test) in Phase 3.

**Risks.**
1. *Retriever ceiling.* If the corpus lacks the supporting passage, re-retrieval cannot repair. The gain may be
   small on long-tail questions (PopQA) and larger on multi-hop. Unknown until measured.
2. *Verifier noise.* FRANQ's long-form detection PR-AUCs are low; GRAB-RAG shows an NLI verifier accepting a wrong
   answer when the model's belief agrees with misleading evidence. A noisy verifier makes repair "launder" errors:
   a wrong claim that gets re-retrieved until something loosely supports it. This must be measured directly
   (precision of repaired claims vs. originally-supported claims).
3. *Guarantee validity.* Split-conformal guarantees assume exchangeable calibration and test data. Repair changes
   the claim distribution, so calibration must be done on the output of the full pipeline, and we should claim
   an empirical target, not a theoretical guarantee, unless we can justify it.
4. *Parametric-confidence signal may be uninformative* at 7–9B (Moskvoretskii et al.: most methods show near-random
   self-knowledge). If so the policy collapses to one axis, which is essentially "re-retrieve unsupported claims".
5. *Decomposition errors* with a 7–9B decomposer (noted as a limitation by Rehman et al.).

**Compute (single RTX 4000 Ada, 20 GB).** Per question: 1 generation + 1 decomposition + k short closed-book
confidence probes (batched) + per-claim NLI on a small encoder + for the repaired subset 1 retrieval and
1 short revision each. Roughly 3–6 generator calls per question plus encoder passes. Feasible with one 7–9B
generator served by vLLM and a small verifier, **if** the verifier fits beside the generator: Phase 0 measured
~1.5 GB VRAM headroom with a BF16 7B model, so this needs either a 4-bit/AWQ generator or running the verifier
on CPU / in a second pass. To be measured in Phase 3, not assumed.

---

## Candidate B — Budgeted verification (cost-aware allocation)

**Idea.** Treat verification and repair as a spend: given a per-question budget (generator tokens, retrieval
calls, GPU-seconds), rank claims by a cheap risk score and spend the expensive actions (re-retrieval, LLM
revision, stronger verifier) where the expected gain is highest. The deliverable is the frontier
*factual precision × claim retention × cost*, and a policy that is better than uniform spending at equal budget.

**Closest prior work and the difference.**
- Moskvoretskii et al. (2501.12835): cost accounting for the *retrieval decision* only.
- Chen et al. (2603.16817): compares verifier FLOPs vs accuracy, but does not allocate a budget across claims.
- EAVer (2609.22223): learned, search-efficient verification for *evaluation*, requires fine-tuning.
- Ciosek et al. (2504.03579): adaptive sample budgets for semantic entropy (detection).
- Su et al. (2502.12145): user-controlled accuracy–cost knob for retrieval.

**Honest novelty rating: low-to-moderate as a standalone method; useful as an evaluation axis.** I found no
budget-allocation policy for claim-level repair, but the idea is a direct transfer of known cascades.

**Risks.**
1. *The budget may not bind.* If a small NLI verifier is cheap enough to check every claim (Chen et al. suggest
   lightweight verifiers are orders of magnitude cheaper), the only expensive actions are re-retrieval and
   revision, and "verify everything, repair the flagged ones" may already be near-optimal. Then B reduces to
   reporting cost honestly — still valuable, but not a method.
2. Needs a risk score that ranks claims well; same signal-quality problem as A.
3. Cost numbers are hardware-specific; must report tokens and calls as well as GPU-seconds.

**Compute.** Same components as A, swept over 4–6 budget levels → roughly 4–6× A's inference cost on dev data.
Caching of generations, retrievals and verifier scores makes most of the sweep free.

---

## Candidate C — Faithfulness-aware calibrated filtering  (lowest risk, incremental)

**Idea.** Keep the filter-only pipeline of conformal factuality, but replace its single claim score with a
two-signal score in the spirit of FRANQ: evidence support for faithful claims, parametric confidence for
unfaithful ones. No repair, no re-retrieval.

**Closest prior work.** Conformal-RAG already uses "internal information from the RAG mechanism" in its score;
FRANQ already defines the two-signal estimator; Chen et al. already compare scoring functions. Two-sided conformal
evidence sets (2611.10335) calibrate support and refutation separately.

**Honest novelty rating: low.** This is a new scoring function inside an existing framework. It would most likely
be judged incremental at a Q1 venue on its own.

**Why list it.** It is cheap, almost certain to be implementable, and it is the natural *ablation* of A
("A without repair"). Its result tells us how much of A's gain, if any, comes from repair rather than scoring.

**Risks.** Small or no gain over existing scorers; FRANQ's signals are noisy on long-form text.

**Compute.** Lowest: 1 generation + decomposition + scoring per question.

---

## Recommendation

**A as the method, B as an evaluation axis inside it, C as an ablation.** Concretely the paper would claim:

1. a repair-before-filter pipeline (A) evaluated against filter-only conformal factuality on retention and
   non-empty rate at matched factuality targets;
2. every result reported with cost (B's accounting), including a "verify all / repair flagged" vs
   "budget-ranked" comparison — reported whichever way it comes out;
3. fair re-runs of the baselines required by the project rules (no-retrieval, standard RAG, and at least three of
   Self-RAG, CRAG, FLARE, CoVe, RARR, SelfCheckGPT-style filtering) on the same retriever, corpus and generators,
   plus conformal filtering as the directly competing baseline. A SeaRAG comparison would be our own
   re-implementation and labelled as such.

**What would make me advise against A.** If a dev-set pilot shows that under 10–15% of dropped claims are
repairable with our retriever, the headline effect cannot be large. In that case the honest paper is the
controlled comparative study (items 2–3) with a negative result on repair — publishable, but a weaker fit for a
"new method" Q1 paper. I recommend making that pilot the first experiment of Phase 5 and fixing the decision rule
in the Phase 3 plan.

## Things I could not verify (carry into Phase 3)

- Whether "repair before conformal filtering" appears in work outside arXiv, or in the conformal-RAG papers'
  appendices beyond the passages I searched (I grepped the full texts for repair/re-retrieval/revision terms and
  read the surrounding lines; I did not read all four end to end).
- 2505.24830 (medical atomic fact-checking with correction): abstract only; it does decompose–verify–correct in a
  domain setting and must be read before the paper's related-work section is written.
- CtrlA, PAIRS, SUGAR, SEReDeEP, BEE-RAG (baselines named in SeaRAG): not looked up.
- Availability, size and licence of candidate verifiers (MiniCheck, AlignScore, TRUE) and whether one fits next to
  a 7–9B generator on 20 GB: to be checked on the HF Hub and measured.
- Venues of all cited papers remain unconfirmed (see `refs/VERIFIED_REFS.md`).
