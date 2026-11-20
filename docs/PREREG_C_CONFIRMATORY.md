# Preregistration — confirmatory test of the two-signal filter score C (H3–H5)

**Written 2026-11-20, before any output exists for the data below.** Frozen by the git commit that adds this file and
pushed to the public code repository (https://github.com/shivamsengar135/repair-vs-filter-rag) before the first
generation call; the push time is the public timestamp. Any later change is listed under "Deviations" at the end,
with its date and reason, and never removes the original text.

## 1. Why this test

In the main study (EXPERIMENT_PLAN.md) the confirmatory hypotheses H1/H2 (repair before filtering, A vs CF) were not
supported. In EXPLORATORY, unadjusted tests on the same test data, the two-signal score C kept significantly more
judge-supported claims than CF at target 0.90 in 5 of 6 generator–dataset pairs (biographies: Qwen2.5-7B +1.82
[1.30, 2.39], Mistral-7B +1.27 [0.96, 1.61], Falcon3-7B −0.21 [−0.47, 0.06]; ASQA: +0.13 to +0.16; ledger row
"Pairwise long-form comparisons", `results/test/exploratory_tests.json`). Because C was singled out after looking at
those results, the finding is tested again here on data that no method has ever been run on.

## 2. Data (fixed)

- **`data/splits/bio_confirm.jsonl`**: the 183 FActScore entities that have human labels
  (`factscore/data/labeled/prompt_entities.txt`), prompt "Question: Tell me a bio of <entity>." — same format as
  bio_dev / bio_test. 183 rows, sha256 `8fb76a61b229f797df8ce3ae751bc8feb6c1dc87a22f6bcafea490fea8c800ff`
  (`scripts/build_confirm_split.py`; MANIFEST entry `bio_confirm`). No overlap with bio_dev / bio_test.
- These entities were excluded from the main splits and used only for judge validation 1, which judged FActScore's
  released answers of InstructGPT, ChatGPT and PerplexityAI — never an answer of our generators or a score of any
  of our methods. FActScore's human labels are NOT used in this test.
- The judge's evidence exists for all 183 (FActScore knowledge-base page; `scripts/check_confirm_kb.py`).
- **ASQA is not part of the test:** all 948 ALCE ASQA items were already used (200 dev, 748 test), so no fresh ASQA
  data exists in our setup. The ASQA part of the exploratory finding stays exploratory.

## 3. Methods and setup (identical to the main study's test runs)

Generators G1 Qwen2.5-7B-Instruct (rev a09a3545…), G2 Mistral-7B-Instruct-v0.3 (rev c170c708…), G3
Falcon3-7B-Instruct (rev 1e57a0ec…), vLLM 0.33.0 in Docker, bfloat16, greedy, ≤ 512 new tokens; BM25 over the DPR
Wikipedia split, k = 5, query = entity name; decomposition prompt v2; MiniCheck-Flan-T5-Large support s; closed-book
confidence c; C score = s + (1 − s)·c; CF score = s; conformal calibration as in the paper. Judge Phi-4 FP8
(`RedHatAI/phi-4-FP8-dynamic`, rev 8f62f4b8…) with FActScore's evidence layout. Code: `scripts/run_longform.py`
(only change: "confirm" added to the allowed split names) and `scripts/run_judge.py`, driven by
`scripts/confirm_runs.sh`. One run per generator; no tuning, no re-runs except to resume an interrupted job.

## 4. Hypotheses (confirmatory)

At target 0.90 (response-level: all kept claims judge-supported with probability ≥ 0.90), on bio_confirm:

- **H3 (G1, Qwen2.5-7B):** C keeps more judge-supported claims per answer than CF.
- **H4 (G2, Mistral-7B):** same.
- **H5 (G3, Falcon3-7B):** same. *Stated expectation: not supported* — in the exploratory data Falcon3's closed-book
  confidence was nearly uninformative (AUROC 0.537) and C did not beat CF. H5 is included so that the test is not
  limited to the generators where C already looked good.

## 5. Test and decision rule

- Endpoint: judge-supported claims per answer.
- Procedure (identical to H1/H2 and to the exploratory C-vs-CF tests; code `scripts/confirmatory_c.py`, which calls
  `compare()` from `scripts/exploratory_tests.py`): calibration/evaluation partition 0 (calibration fraction 0.5,
  seed 0), paired bootstrap over the evaluation answers (10,000 resamples, seed 0), two-sided p-value.
- Multiplicity: Holm over {H3, H4, H5} at α = 0.05.
- **A hypothesis is SUPPORTED iff Holm rejects it and the difference C − CF is positive.**
- Code check (done before this file was committed): on the existing TEST runs the script reproduces the published
  exploratory numbers exactly (+1.823 [1.297, 2.394]; +1.274 [0.960, 1.611]; −0.206 [−0.469, 0.057];
  `results/test/confirmatory_c_codecheck.json`).

### Secondary (reported, not part of the decision)
- Precision non-inferiority: one-sided 95% lower bound of the C − CF micro precision difference > −0.02 (the H1/H2
  margin). **Why it is not part of the decision rule (unlike H1/H2):** a power simulation on the test data
  (`scripts/power_confirm_c.py`, `results/test/power_confirm_c.json`; 400 random subsets of 183 of the 350 test
  biographies) gives, at the strictest Holm level α/3:

  | Generator | P(significant, C > CF) | P(precision guard holds) | P(both) |
  |---|---|---|---|
  | Qwen2.5-7B | 0.87 | 0.51 | 0.41 |
  | Mistral-7B | 0.81 | 0.60 | 0.47 |
  | Falcon3-7B | 0.19 | 0.17 | 0.01 |

  With 183 biographies the guard's bound is too wide to hold reliably even where precision is not lower on the full
  test data (G1 bound −0.019 at n = 350), so requiring it would make the decision mostly noise. It is reported for
  every hypothesis with its bound and point estimate.
- Descriptive: the mean, standard deviation and share > 0 of C − CF over the 100 partitions; C − CF at targets 0.80,
  0.85, 0.95; AUROC of s, c and C against the judge labels; the judge-supported rate.

## 6. What will be reported

All three results, whatever they are, in the paper and in `results/RESULTS_LEDGER.md`, labelled as a confirmatory
test preregistered after the main study, on fresh biographies only. A non-supported H3 or H4 is reported as such, and
the exploratory claim about C is then weakened in the paper accordingly. The FActScore human labels of these
entities are not used, and no other analysis on bio_confirm is called confirmatory.

## 7. Deviations

- **2026-11-20, after the analysis ran (13:05Z) — decision field of the script.** `scripts/confirmatory_c.py` (written
  before section 5 was finalised) sets its `SUPPORTED` field as "Holm rejects AND difference > 0 AND precision guard
  holds" — the H1/H2 rule. Section 5 of this preregistration, committed and pushed before any output existed,
  makes the guard secondary and defines SUPPORTED as "Holm rejects AND difference > 0". The preregistered rule is
  applied; the script output is left unchanged (`results/confirm/confirmatory_c.json`, which contains every quantity
  needed for both rules) and both readings are reported: H3 is supported under section 5 and would not be under the
  H1/H2 rule (its guard bound is −0.063); H4 and H5 are the same under both.

## 8. Result (added 2026-11-20 after the single analysis run; nothing above was changed)

Run log `logs/confirm_runs.log` (CONFIRM_START 11:36:51Z at commit df1568c, CONFIRM_END 13:05:36Z, exit 0); 183
answers per generator, no unreadable judge verdicts, every claim with a knowledge-base page. Evaluation half n = 91.

| | C − CF supported claims / answer [95% CI] | p | Holm | Decision (section 5) | Precision diff (one-sided 95% bound) |
|---|---|---|---|---|---|
| H3 Qwen2.5-7B | +1.077 [+0.549, +1.670] | < 0.0001 | reject | **SUPPORTED** | +0.029 (−0.063, guard fails) |
| H4 Mistral-7B | +0.538 [+0.242, +0.846] | 0.0002 | reject | **SUPPORTED** | +0.148 (−0.000, guard holds) |
| H5 Falcon3-7B | −0.088 [−0.451, +0.297] | 0.668 | not rejected | not supported (as predicted) | +0.109 (+0.025, guard holds) |
