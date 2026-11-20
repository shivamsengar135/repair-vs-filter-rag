# Literature review — hallucination mitigation with retrieval and verification

_Phase 1 output. Compiled 2026-11-12. Every paper named here has a row in `refs/VERIFIED_REFS.md`._

## How to read this document

- **Provenance.** Titles, authors and dates come from the arXiv API or Crossref (`refs/raw/*.json`).
  Numbers come from the papers' own tables, read from arXiv HTML full text saved locally
  (`refs/raw/fulltext/`, not committed). Nothing here is quoted from memory.
- **Read depth.** **[F]** = I read the main results table(s) in the full text. **[A]** = abstract only;
  statements about [A] papers paraphrase the authors' abstract and are **their claims, unchecked**.
  **[T]** = only the title and metadata were verified; nothing about the content is asserted.
- **All numbers below are "reported in the original paper (different setup)".** Models, retrievers, corpora
  and metrics differ between papers. They must never be put in a table next to our own results.
- **Venues are unconfirmed** (author-declared arXiv comment / Semantic Scholar only). Semantic Scholar's
  venue field was visibly wrong for at least two entries (CRAG, RARR), so confirm every venue on the
  publisher page before writing BibTeX.
- **Code** = a repository link is printed in the paper or its arXiv record. I did not open or run any repository.

## 1. Comparison table — mitigation methods

| Method | Core idea | Datasets | Metrics | Base model(s) | Reported numbers (original setup) | Code link in paper | Limitations (L = stated by authors and read by me; M = my assessment) |
|---|---|---|---|---|---|---|---|
| **Self-RAG** (Asai et al.) [F] | Fine-tune the generator to emit reflection tokens: retrieve on demand, then critique its own passages and output. | PopQA, TriviaQA, PubHealth, ARC-Challenge, biographies, ASQA | accuracy, FActScore, str-EM, ROUGE, MAUVE, citation precision/recall | Llama-2 7B and 13B, fine-tuned | 7B: PopQA 54.9, TQA 66.4, PubHealth 72.4, ARC 67.3, Bio FActScore 81.2, ASQA citation prec/rec 66.9/67.8. 13B: 55.8 / 69.3 / 74.5 / 73.1 / 80.2, citation 70.3/71.3. Baseline Llama2-7B + retrieval: PopQA 38.2, Bio 78.0. | selfrag.github.io | M: needs generator fine-tuning, so it is tied to one base model and cannot be applied to a new generator without retraining. M: retrieval for Bio and open-domain QA adds web-search documents, which is hard to reproduce. |
| **CRAG** (Yan et al.) [F] | A small T5 "retrieval evaluator" scores retrieved documents and triggers Correct / Incorrect / Ambiguous actions; refine documents or fall back to web search. | PopQA, biographies, PubHealth, ARC-Challenge | accuracy, FActScore | LLaMA2-hf-7b; SelfRAG-LLaMA2-7b | With SelfRAG-LLaMA2-7b: RAG 52.8 → CRAG 59.8 (PopQA); Self-CRAG 61.8 PopQA, 86.2 Bio. With LLaMA2-hf-7b: RAG 50.5 → CRAG 54.9 (PopQA). Evaluator accuracy 84.3 vs ChatGPT 58.0. Time per instance: RAG 0.363 s, CRAG 0.512 s, Self-RAG 0.741 s. | github.com/HuskyInSalt/CRAG | L: "fine-tuning an external retrieval evaluator is inevitable". M: the fallback is live web search (not reproducible); it corrects the *retrieval*, and never checks the generated answer. |
| **FLARE** (Jiang et al.) [F] | Generate a tentative next sentence; if it contains low-probability tokens, use it as a query, retrieve, and regenerate. | 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp | EM, F1, Disambig-F1, ROUGE-L | text-davinci-003 | 2Wiki EM: no retrieval 28.2, single retrieval 39.4, FLARE 51.0 (F1 59.7). | github.com/jzbjyb/FLARE | M: needs token probabilities; tuned on a closed API model that is no longer the norm. In DRAGIN's re-implementation on Llama-2/Vicuna it is often *below* single-round RAG (see DRAGIN row). |
| **DRAGIN** (Su et al.) [F] | Trigger retrieval from token uncertainty weighted by attention importance; build the query from attention over the whole context. | 2WikiMultihopQA, HotpotQA, StrategyQA, IIRC | EM, F1, accuracy | Llama2-7b/13b-chat, Vicuna-13b; BM25 over Wikipedia (100-token passages) | Llama2-13b 2Wiki EM: no-RAG 0.187, single-round 0.245, FLARE 0.224, DRAGIN 0.304. HotpotQA EM: 0.223 / 0.263 / 0.180 / 0.314. Retrieval calls per question: 2.6–4.8. | github.com/oneal2000/DRAGIN | M: white-box access to attention is required; several retrieval rounds per question; no verification of the final answer. |
| **Adaptive-RAG** (Jeong et al.) [F] | A small classifier predicts query complexity and routes to no-retrieval, single-step or multi-step RAG. | SQuAD v1.1, Natural Questions, TriviaQA, MuSiQue, HotpotQA, 2WikiMultiHopQA (averaged) | EM, F1, accuracy, steps, time | FLAN-T5-XL/XXL, GPT-3.5 | FLAN-T5-XL average: single-step F1 44.31 (1.00 time), multi-step 48.85 (8.81× time), Adaptive-RAG 46.94 (3.60× time), oracle router 56.28. | github.com/starsuzi/Adaptive-RAG | L: classifier labels are assigned automatically and may be noisy. M: the large oracle gap shows routing is far from solved. |
| **SeaKR** (Yao et al.) [A] | Retrieve when the LLM's internal-state uncertainty is high; re-rank passages by how much they reduce it. | "complex and simple QA" | — | — | not extracted | github.com/THU-KEG/SeaKR | M: needs hidden states (white-box). |
| **CoVe** (Dhuliawala et al.) [F] | Draft → plan verification questions → answer them independently → revise. No retrieval. | Wikidata / Wiki-Category lists, MultiSpanQA, biographies | precision, F1, FActScore | Llama 65B | Biographies FActScore: few-shot 55.9 → CoVe factor+revise 71.4 (average facts 16.6 → 12.3). MultiSpanQA F1 0.39 → 0.48. | none found in paper | M: verifies against the model's own parametric knowledge, so it cannot fix what the model does not know; the gain comes with fewer facts per answer; 65B model. |
| **RARR** (Gao et al.) [F, partial] | Post-hoc: generate questions about a finished text, retrieve evidence, edit unsupported parts while preserving the rest. | NQ statements, StrategyQA chains, QReCC dialogs | attribution (AIS, auto-AIS) before→after, preservation | texts generated by PaLM and LaMDA | StrategyQA row for RARR: attribution 37.6 → 45.1 (first attribution column), 24.5 → 31.5 (second), preservation columns 92.6 / 89.9 / 84.6. QReCC: 18.8 → 29.4. Column-to-metric mapping not double-checked; NQ row not extracted. | github.com/anthonywchen/RARR (not checked whether official) | M: attribution gains are modest in absolute terms; evaluated on outputs of large proprietary models. |
| **Chain-of-Note** (Yu et al.) [A] | Write sequential reading notes on retrieved documents to judge relevance before answering; answer "unknown" when knowledge is lacking. | four open-domain QA benchmarks | EM | LLaMa-2 7B trained on ChatGPT-created data | "+7.9 in EM" on average (condition truncated in what I read) | not checked | M: requires fine-tuning. |
| **Astute RAG** (Wang et al.) [A] | Elicit the LLM's internal knowledge, consolidate it with retrieved passages with source-awareness, answer by information reliability. | — | — | Gemini, Claude | not extracted | not checked | M: evaluated on proprietary models only (per abstract). |
| **InstructRAG** (Wei et al.) [A] | The LM explains how the gold answer follows from retrieved documents; these rationales serve as in-context demonstrations or fine-tuning data. | five knowledge-intensive benchmarks | — | — | "relative improvement of 8.3% over the best baseline" on average | github.com/weizhepei/InstructRAG | M: needs gold answers to synthesise rationales. |
| **R-Tuning** (Zhang et al.) [A] | Refusal-aware instruction tuning: train the model to say "I don't know" outside its parametric knowledge. | — | — | — | not extracted | github.com/shizhediao/R-Tuning | M: requires fine-tuning; not about retrieved evidence. |

## 2. Comparison table — detection, verification and evaluation

| Work | What it provides | Reported numbers (original setup) | Code link in paper | Relevance to us |
|---|---|---|---|---|
| **SelfCheckGPT** (Manakul et al.) [F] | Black-box detection: sample several answers and score each sentence's consistency with the samples. Data: GPT-3 passages about WikiBio entities. | Sentence-level AUC-PR (NonFact): NLI variant 92.50, Prompt variant 93.42, vs GPT-3 token probabilities 83.21–87.51. | github.com/potsawee/selfcheckgpt | Candidate baseline (filtering). Cost: N extra generations per answer. |
| **Semantic entropy** (Kuhn et al.) [F, partial]; Nature version (Farquhar et al., doi:10.1038/s41586-024-07421-0, metadata only) | Entropy over meaning-clusters of sampled answers as an uncertainty signal. | Abstract: more predictive of QA accuracy than comparable baselines. Tables not extracted. | github.com/lorenzkuhn/semantic_uncertainty | Candidate abstention / retrieval-trigger signal. |
| **Semantic Entropy Probes** (Kossen et al.) [A] | Approximate semantic entropy from hidden states of a single generation. | Abstract: SE costs a "5-to-10-fold increase in computation"; probes avoid sampling. | not checked | Evidence that sampling cost is a recognised problem. |
| **FActScore** (Min et al.) [F] | Decompose a biography into atomic facts; score the fraction supported by Wikipedia. | Human-judged FActScore: InstructGPT 42.5, ChatGPT 58.3, PerplexityAI 71.5. Automatic estimator "<2% error rate" (authors' claim). | github.com/shmsw25/FActScore | Our long-form metric. Note the original estimator used ChatGPT/Inst-LLaMA as judge. |
| **SAFE / LongFact** (Wei et al.) [F, partial] | LLM agent splits a response into facts and checks each with Google Search. | Agrees with crowd annotators on 72% of ~16k facts; wins 76% of 100 sampled disagreements; ">20 times cheaper" than humans. | github.com/google-deepmind/long-form-factuality | Needs a search API and a strong LLM — not runnable as-is on our hardware. |
| **VeriScore** (Song et al.) [A] | Like FActScore/SAFE but only scores *verifiable* claims. | not extracted | not checked | Alternative long-form metric. |
| **MiniCheck** (Tang et al.) [F, partial] | Small fact-checkers trained on synthetic data; LLM-AggreFact benchmark. | Abstract: "GPT-4-level performance but for 400x lower cost". Table 2 (balanced accuracy, average): GPT-4 75.3, AlignScore and SummaC-ZS lower (67.9 for SummaC-ZS). MiniCheck's own row not extracted. | not extracted | Candidate cheap claim-vs-evidence verifier. |
| **SummaC**, **AlignScore**, **WiCE**, **Molecular Facts** [T] | Going by titles only: NLI-based inconsistency detection; a unified alignment function for factual consistency; entailment for claims in Wikipedia; decontextualisation in LLM fact verification. | not read | SummaC: github.com/tingofurro/summac | Candidate verifiers and claim-decomposition issues — must be read before use. |
| **ALCE** (Gao et al.) [F, partial] | Benchmark and automatic metrics (fluency, correctness, citation quality) for answers with citations. Includes ASQA and ELI5 subsets (per VeriCite's setup section); full dataset list not checked. | On ELI5, "even the best models lack complete citation support 50% of the time". | github.com/princeton-nlp/ALCE | Our citation precision/recall metric and data. |
| **AIS** (Rashkin et al.), **Attributed QA** (Bohnet et al.) [T] | Going by titles only: measuring attribution in generation; evaluation and modelling for attributed QA. | not read | not checked | Likely the definitions behind citation metrics — must be read before use. |
| **RAGTruth** (Niu et al.) [F] | "Nearly 18,000" naturally generated RAG responses with manual case-level and word-level hallucination annotations (QA, data-to-text, summarisation). | Response-level F1 overall: GPT-4-turbo prompt 63.4, SelfCheckGPT (GPT-3.5) 58.8, fine-tuned Llama-2-13B 78.7. | not extracted | Our detection benchmark. Fine-tuned detectors beat prompted GPT-4. |
| **HaluEval** (Li et al.) [A] | Generated and human-annotated hallucinated samples for recognition tests. | not extracted | github.com/RUCAIBox/HaluEval | Alternative detection benchmark (ChatGPT-generated plus human-annotated samples). |
| **FAVA** (Mishra et al.) [A] | Fine-grained hallucination taxonomy, benchmark and a retrieval-augmented detector/editor. | not extracted | fine-grained-hallucination.github.io | Related detect-and-edit system. |
| **PopQA** (Mallen et al.) [A] | 14k-question open-domain QA set. Abstract: LMs struggle with less popular knowledge and scaling does not fix the long tail; retrieval-augmented LMs "largely outperform orders of magnitude larger LMs". | not extracted | github.com/AlexTMallen/adaptive-retrieval | Our short-form dataset; origin of popularity-based adaptive retrieval. |
| **ClashEval** (Wu et al.) [A] | 1,200+ questions testing whether a model follows wrong retrieved content over its own prior. | not extracted | not checked | Motivates checking retrieved evidence, not just trusting it. |

## 3. Recent work, 2025–2026

Found by ten arXiv API queries (`refs/search_queries.txt`, 131 unique hits, `logs/phase1_arxiv_search.log`) plus two
web searches. 34 were shortlisted and their abstracts read. All are arXiv preprints unless noted; **peer-review
status was not checked**. Entries are the authors' own claims.

### 3a. Closest to a "verify claims against evidence, then act" pipeline

| Paper | What it does (from abstract; [F] where I read further) | How close to our candidate ideas |
|---|---|---|
| **SURE-RAG** (Qiu et al., 2605.03534) [F] | Evidence-*sufficiency* verification for selective answering. A fine-tuned DeBERTa-v3-base cross-encoder scores each (claim, passage) pair; a logistic-regression aggregator turns these into a Supported / Refuted / Insufficient label and a selective score. Benchmark "HotpotQA-RAG v3": 4,026 examples built by removing evidence or perturbing the answer. Macro-F1 0.8951 ± 0.0069 (3 seeds), 0.9075 calibrated, vs mean-pooling 0.6516 and a concat cross-encoder 0.8888. Risk@30% coverage 0.1642 vs 0.2588, but Risk@70 is 0.5457 (no better than pooling). | **Close** to "claim verification + abstention", but narrower than it looked. Confirmed by full read: (1) **no generator** — candidate answers are gold or perturbed, not model output; (2) **no re-retrieval and no revision**; (3) short answers only, "long-form multi-claim coverage is future work"; (4) verifier is trained on the benchmark's own construction labels; (5) authors: controlled benchmark, "not a complete solution to open-world hallucination detection"; GPT-4o beats it on natural HaluBench. |
| **GRAB-RAG** (Setiawan et al., 2608.22228) [F] | Benchmark: same questions under supportive / degraded / missing / misleading context, on three frozen 3.8–8B models. Prompted abstention still answers 41.6% of misleading-context questions. A conflict check cuts this to 13.3% but abstains on 49.8% of the questions the model could otherwise answer (their "capable set"); an NLI verifier reaches 14.0% with 31.0%. | **Direct evidence about our model size class.** Shows NLI verification helps but fails when parametric knowledge agrees with the planted error. |
| **VeriCite** (Qian et al., 2510.11394) [F] | Three stages: initial answer with NLI-checked citations; evidence extraction from each passage, NLI-verified; LLM refinement that reorders and merges without changing content. NLI model: TRUE. Top-5 passages, greedy decoding, 2 few-shot examples. ASQA, ELI5, HotpotQA, MuSiQue; Llama3-8B, Gemma-2-9B, GLM-4-9B, Qwen2.5-7B and 14B. ASQA ablation (Llama3-8B): removing NLI leaves correctness unchanged (41.63 → 41.59) but cuts citation F1 77.73 → 68.91. Using the generator itself as verifier is worse than NLI (citation F1 73.01 vs 80.05 on 200 samples). Authors report it *under*performs strong baselines on ELI5 correctness and on multi-hop with 3 of 4 models. Code: github.com/QianHaosheng/VeriCite. | **Close** on the citation side and on our exact model class. Not covered: abstention, re-retrieval, hallucination rate, or any cost/latency measurement (none found in the paper). Useful prior: a small NLI model beat an 8B LLM as verifier. |
| **RT4CHART** (Yu et al., 2603.27752) [F] | Detector only: sentence-level claim decomposition, chunk-local then global verification with GPT-4o mini, labels entailed / contradicted / baseless, maps to answer spans. Answer-level F1 0.776 on RAGTruth++ (Lettuce 0.371, Vectara 0.424) and 0.845 on RAGTruth-Enhance (Vectara 0.724). 3 repeated runs: 0.779 ± 0.014. Cost $0.0104 per sample. | No mitigation. Judged strictly against the given context (authors: it "may flag factually correct statements as baseless"). **Important for our dataset choice:** re-annotating a 408-example RAGTruth subset raised hallucination spans from 86 to 865, i.e. the original RAGTruth labels under-count badly. |
| **FRANQ** (Fadeeva et al., 2505.21072) [F] | Detector only. P(claim true) = P(faithful)·P(true | faithful) + P(unfaithful)·P(true | unfaithful), with AlignScore for faithfulness and different uncertainty estimators per branch. New long-form set: 76 questions, 1,782 / 1,548 atomic claims, labelled for both faithfulness and factuality (GPT-4o-search first pass, manual review of False/Unverifiable; annotation accuracy 78% → 91%). Models: Llama 3B/8B, Falcon 3B, Gemma 4B/12B; Contriever over 2018 Wikipedia. Long-form PR-AUC values are low in absolute terms (general baselines ≈ 0.05–0.16). Overhead 1.7 s per instance on top of a 2.2 s generation. Code: github.com/stat-ml/rag_uncertainty. | Establishes the faithful-vs-factual split for *detection*, on small open models. The authors name "using franq's uncertainty signals for generation-time control" as future work — i.e. mitigation driven by this split is, by their account, open. Also notes RAGTruth treats correct-but-unsupported content as hallucination. |
| **UniCR** (Oehri et al., 2509.01455) [F] | Describes fusing likelihood, self-consistency, retrieval compatibility (claim coverage, max entailment, contradiction) and verifier feedback into a calibrated correctness probability, with refusal under conformal risk control. | The *idea* "calibrated abstention from multiple evidence signals incl. claim-level entailment" is written down here. **But I would not rely on its results:** the text has no results tables, names no models or datasets specifically ("open LLM backends", "standard corpora"), gives outcomes only as ranges ("lowers AURC by 12–25%"), and the arXiv HTML still carries template placeholders (author e-mails at `example.li` / `example.sm`, "Woodstock, NY", ISBN "978-1-4503-XXXX-X"). Cite, if at all, only as a related proposal. |
| **CoVe-RAG+** (Joseph et al., 2609.26229) [A] | CoVe plus RAG with iterative self-verification, for engineering/CAD documents. | Shows "CoVe + retrieval" is already published as an idea, in a narrow domain. |
| **Atomic fact-checking in medical RAG** (Vladika et al., 2505.24830) [A] | Decompose long answers into atomic facts, verify each against guidelines, correct errors. | Decompose–verify–correct already exists in a domain setting. |
| **ArbGraph** (Niu et al., 2604.18362) [A] | Decompose *retrieved documents* into claims, build a support/contradiction graph, arbitrate before generation. | Verification moved before generation. |
| **Reason and Verify** (Khan et al., 2603.10143) [A] | Rationale generation grounded in evidence spans plus a verification taxonomy; BioASQ, PubMedQA. | Domain-specific neighbour. |
| **SeaRAG** (Yuan et al., WWW 2026, doi:10.1145/3774904.3792598) [F — all 10 pages incl. appendix, from the PDF the user supplied] | Sentence-by-sentence loop. After each generated sentence the *same LLM* is asked verification questions about its entities and about the whole statement; entropy, max-probability and uncertainty words in its reply form a hallucination score. If the score passes a threshold: retrieve 6 passages, rank them by the entropy of passage-conditioned probes, regenerate the sentence with the top 3. Models: Mistral-7B, Llama-3-8B, Qwen-2.5-7B. Table 1 (Llama-3-8B, accuracy): always-retrieve vs SeaRAG w/ SC — HotpotQA 27.20 / 36.30, 2Wiki 32.21 / 39.15, NQ 25.13 / 33.45, PopQA 31.04 / 35.96, TriviaQA 55.59 / 66.71; FLARE 59.94 on TriviaQA. Retrieval rate 45.2%. Overhead +36.14% time (7,560 → 10,760 ms); "at most 3" LLM calls per sentence. Thresholds grid-searched on 5% of TriviaQA. | **Same family as our ideas (training-free detect → retrieve → regenerate) on our model class, so it must be cited and, if possible, compared against.** What it does *not* do (confirmed): detection is self-probing of parametric confidence, **not** verification of claims against retrieved evidence; the regenerated sentence is not re-checked; **no abstention, no citation, no long-form factuality metric**; only short-answer QA accuracy. **Reproducibility concerns (my reading):** no code link and no limitations section; retriever and corpus are never named; entity extraction uses an unnamed "frontier LLM augmented with web search", which sits oddly with "training-free" and with the cost table; the setup text lists TriviaQA, PopQA and FreshQA while the tables report six other-or-overlapping datasets and no FreshQA; the ablation rows (temperature, top-k, top-p, "last layer only") and the conclusion's "hidden-state probes" do not match the method section, which describes output-token entropy; "averaged over 5 runs" but no variance is shown; base memory 37.6 GB for a 7–8B model is unexplained. Reproducing it as a baseline would mean re-implementing from the description, and the result should be labelled as our re-implementation. |

### 3b. Adaptive retrieval and its cost

| Paper | Finding (authors' claim) |
|---|---|
| **Moskvoretskii et al., 2501.12835** [F] | 35 adaptive-retrieval methods (8 pipelines, 27 uncertainty estimators), LLaMA 3.1-8B-instruct, BM25 over Wikipedia, 500 questions from each of 6 datasets. In-accuracy, Never / Always RAG: NQ 0.446 / 0.496, HotpotQA 0.286 / 0.410, 2Wiki 0.318 / 0.374, MuSiQue 0.106 / 0.100. Simple uncertainty triggers reach NQ 0.512 with ~1.6–1.8 LM calls; DRAGIN 0.480 with 4.5; Rowen variants need 29–80 LM calls per question; SeaKR 10–15. Multi-step methods help on 2Wiki (IRCoT / Adaptive-RAG / DRAGIN ≈ 0.454–0.456) at 4–6 LM calls. "Most adaptive retrieval methods may lack the ability to accurately identify self-knowledge"; self-knowledge degrades out-of-domain. Limitation (authors): one model only. Code: github.com/s-nlp/AdaRAGUE. |
| **Marina et al., 2505.04253** [A] | Retrieval decisions from question-only features (27 features) match LLM-based methods at much lower cost. |
| **Su et al., 2502.12145** [A] | Two classifiers plus a user parameter α trade accuracy against retrieval cost. |
| **DeepRAG, 2502.01142** [A] | Retrieval decisions as an MDP over decomposed sub-queries. |
| **Know3-RAG, 2505.12662** [A] | Knowledge-graph signals drive adaptive retrieval, reference generation and filtering. |
| **DeclineCraft, 2610.28340** [A] | Across 27 LLMs: single-shot retrieval raises correct abstention, but *iterative* RAG lowers it in 30 of 40 configurations with little accuracy change. |

### 3c. Abstention, uncertainty, detection, attribution

- **Abstention by training:** Divide-Then-Align (2505.20871), ERA (2604.20854), CalibraRL (2610.35142) — all fine-tune or use RL. **Sieve and Sage** (2609.35794) screens retrieved sets for distracting evidence with a light module before the LLM. [A]
- **Uncertainty:** Bayesian semantic-entropy estimation needing 53% of the samples (2504.03579); Semantic Energy (2508.14496); R2C for multi-step retrieval-augmented reasoning (2510.11483). [A]
- **RAG hallucination detectors:** LettuceDetect (2502.17125; ModernBERT token classifier, 79.22% example-level F1 on RAGTruth per abstract), LUMINA (2509.21875), FaithLens (2512.20182; 8B model with explanations), evidence-aligned entity verification (2609.08267), FaithJudge (2505.04847). [A]
- **Attribution:** generation-time vs post-hoc citation (2509.21557) reports a consistent coverage–correctness trade-off; the attribution–compression frontier (2609.14245) reports a compressor whose citations score 0.86 precision against its summaries but 0.12 against source spans. [A]
- **Long-form evaluation:** VeriFastScore (2505.16973; fine-tuned Llama3.1 8B, addresses >100 s per response for FActScore/VeriScore), FactReasoner (2502.18573). [A]

## 4. Gap analysis

Each gap lists the evidence for it and how sure I am. "Not found" means not found by the searches above — it is
not proof of absence.

_Revised 2026-11-12 after reading SURE-RAG, UniCR, VeriCite, RT4CHART, FRANQ and Moskvoretskii et al. in full
(SeaRAG: abstract only). Changes from the first pass are marked "after full read"._

**G1. The cost of verification-based *mitigation* is not measured against its benefit.** *Confidence: moderate.*
After full read, cost *is* measured in two neighbouring places, so the gap is narrower than first stated:
(a) for the retrieval decision — Moskvoretskii et al. count LM and retriever calls for 35 methods and find some
pipelines need 29–80 LM calls per question for no accuracy gain; (b) for detectors — FRANQ (1.7 s on top of a
2.2 s generation) and RT4CHART ($0.0104 per sample). CRAG reports time per instance. What I still did not find:
accuracy, hallucination rate and coverage reported as a function of verification budget (tokens, calls,
GPU-seconds) for a decompose → verify → re-retrieve/revise/abstain pipeline on a fixed open 7–9B generator.
VeriCite, the closest pipeline on our model class, reports no cost at all. Our single-GPU setting makes this
concrete: a BF16 7B generator leaves no VRAM for a second model (Phase 0).

**G2. Components are evaluated in isolation; the full chain is not.** *Confidence: moderate-to-high after full read.*
- SURE-RAG has no generator (answers are gold or perturbed), no re-retrieval, no revision, short answers only.
- RT4CHART and FRANQ are detectors; neither changes the output.
- VeriCite verifies and rewrites for citation quality, with no abstention or re-retrieval, and loses correctness
  on ELI5 and on multi-hop for most models.
- Moskvoretskii et al. compare retrieval triggers only.
- SeaRAG (read in full) is the closest: training-free detect → retrieve → rank → regenerate on 7–8B models.
  But its detector is the model's own confidence, not a check against evidence; nothing verifies the regenerated
  text; there is no abstention; and it is evaluated only on short-answer accuracy. It narrows this gap for
  *uncertainty-triggered retrieval and regeneration* and leaves *evidence-grounded claim verification with
  differentiated actions* open.
A controlled study of which stages pay for themselves under one retriever, corpus and generator still appears
to be missing. **Honest framing for Phase 2:** a method here would be a recombination of known parts
(decomposition, NLI/LLM verification, re-retrieval, revision, abstention). Its claim to novelty would rest on the
evidence-grounded decision policy and on the controlled cost–benefit evaluation, not on any single component.

**G3. Acting differently on "unfaithful" and "false" is open.** *Confidence: moderate-to-high after full read.*
FRANQ establishes the split for detection and explicitly lists using its signals "for generation-time control"
as future work. RT4CHART's authors state their context-only judgement flags correct statements as baseless.
GRAB-RAG shows an NLI verifier accepting wrong answers when the model's own belief agrees with misleading
evidence. UniCR proposes fusing such signals for refusal, but its evidence is too weak to count (see its row).
A mitigation policy that maps {faithful, unfaithful-but-likely-true, unfaithful-and-likely-false} to
{keep, keep-without-citation or re-retrieve, revise or abstain} was not found. Caveat: FRANQ's long-form PR-AUC
values are low in absolute terms, so the signals this would depend on are noisy.

**G4. Training-free abstention on small frozen models has a poor risk–coverage trade-off.** *Confidence: moderate.*
DTA, ERA, CalibraRL and R-Tuning fine-tune. Without training: GRAB-RAG's best policies cut unsafe answers to
~13–14% while abstaining on 31–50% of the questions the model could otherwise answer; SURE-RAG's advantage exists
only at 30% coverage and vanishes by 70%; Moskvoretskii et al. find most adaptive methods have near-random
self-knowledge and that it degrades out-of-domain. A better curve would be a real contribution — and the
evidence says it is hard.

**Expectation-setting from the full read (not a gap).** On LLaMA 3.1-8B with BM25, always-retrieve vs the best
adaptive trigger differs by under 2 points on NQ (0.496 vs 0.512), and multi-step retrieval gains ~8 points on
2Wiki at 4–6× the LM calls. Large headline gains for a retrieval-side method on current 8B models are unlikely;
the room is in verification, abstention and cost.

**Dataset warning from the full read.** RAGTruth's original span labels under-count hallucinations (86 → 865
spans on a re-annotated 408-example subset, per RT4CHART citing RAGTruth++), and RAGTruth counts
correct-but-unsupported content as hallucination (per FRANQ). If we use RAGTruth in Phase 3 we should use a
re-annotated version and say which, and we should not treat it as a factuality benchmark.

**G5. Reported comparisons are not on common ground.** *Confidence: high — directly observed in the tables.*
- FLARE reports 51.0 EM on 2Wiki with text-davinci-003, yet in DRAGIN's re-implementation on Llama-2/Vicuna it
  falls below single-round RAG on several datasets (e.g. Llama2-13b HotpotQA EM 0.180 vs 0.263).
- CRAG's table shows Self-RAG's decoding applied to plain LLaMA2-hf-7b scoring 29.0 on PopQA against 50.5 for
  standard RAG — the method does not transfer without its fine-tuned checkpoint.
- Adaptive-RAG evaluates Self-RAG with a different base LLM than its own method (acknowledged in its caption).
- Self-RAG and CRAG add web-search results for some tasks.
A fair re-evaluation of these baselines on current open 7–9B models with one retriever and corpus is itself
useful, and is what rule 5 of this project already requires.

**G6. Evaluation depends on API models.** *Confidence: high.*
FActScore's estimator, SAFE and RAGTruth's strongest prompted baselines use ChatGPT/GPT-4 or Google Search.
VeriFastScore and MiniCheck show the field moving to small local evaluators. Our judge will be local and therefore
needs the planned human validation (Cohen's κ).

### What is *not* a gap (ideas that are already published)

- "Combine CoVe with retrieval" — CoVe-RAG+ (2609.26229), and RARR is retrieval-based revision from 2022.
- "Decompose into atomic claims, verify each, fix the unsupported ones" — RARR, FAVA, 2505.24830, RT4CHART.
- "Decide adaptively whether to retrieve" — a crowded area, and 2501.12835 reports that simple uncertainty
  baselines match complex pipelines. A new retrieval trigger alone would be incremental.
- "NLI verifier for citations" — VeriCite, on the same model class we would use.

## 5. Known holes in this review (to close before or during Phase 2)

1. **Still abstract-only:** SeaKR, Astute RAG, Chain-of-Note, InstructRAG, CoVe-RAG+, the medical atomic
   fact-checking paper (2505.24830), ArbGraph. The last three are the nearest remaining unread neighbours.
   (Read in full on 2026-11-12: SURE-RAG, UniCR, VeriCite, RT4CHART, FRANQ, Moskvoretskii et al., and SeaRAG from
   the user-supplied PDF.) Papers SeaRAG cites that I have not looked up: CtrlA, PAIRS, SUGAR, SEReDeEP, BEE-RAG.
2. **Tables not extracted:** semantic entropy (Kuhn/Farquhar), ALCE, MiniCheck's own row, RARR NQ row, HaluEval.
3. **Search coverage:** arXiv only, ten queries, 20 hits each, plus two web searches. No ACL Anthology, OpenReview
   or Google Scholar citation-chasing yet. Peer-reviewed 2025–2026 papers without arXiv versions may be missing.
4. **Venues unconfirmed** for every entry; **code repositories unopened**.
5. **Topics from the brief with thin coverage:** attribution evaluation beyond ALCE/AIS; uncertainty calibration
   (ECE) literature; multi-hop datasets (HotpotQA/2Wiki/MuSiQue papers themselves are not yet in VERIFIED_REFS).
