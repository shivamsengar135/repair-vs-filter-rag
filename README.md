# repair-vs-filter-rag

Research code for a preregistered, controlled comparison of retrieval, verification and conformal claim
filtering for hallucination mitigation in retrieval-augmented LLMs. The main question: does repairing doubtful claims
(re-retrieval or revision) before conformal filtering keep more verifiably correct content than filtering alone?

Short answer from our runs: no. Neither confirmatory hypothesis is supported. The filter score matters more than
repair (exploratory). Every number is traceable through `results/RESULTS_LEDGER.md`.

## Contents

| Path | What it holds |
|---|---|
| `EXPERIMENT_PLAN.md` | Preregistered plan: hypotheses, metrics, statistical tests, decision rules |
| `METHOD_PROPOSALS.md` | Method proposals considered before the plan was fixed |
| `src/hallu/` | Library: retrieval (BM25 / dense), generation client, claim parsing, verifier, conformal filtering, FLARE, metrics, analysis |
| `scripts/` | Entry points: data splits, indexes, runs (`run_longform.py`, `run_shortform.py`, `run_cove.py`, `run_rarr.py`, `run_flare.py`, `run_selfcheck.py`, `run_judge.py`), analysis (`confirmatory.py`, `exploratory_tests.py`, `claim_auroc.py`, `figures.py`) and the phase drivers (`phase5_all.sh`, `phase6_runs.sh`) |
| `configs/prompts/` | All prompts, versioned |
| `data/splits/` | Fixed seeded dev/test splits used in every run |
| `results/` | Per-run outputs (`items.jsonl`), analysis JSON, figures, and `RESULTS_LEDGER.md` (number → script, config, seed, commit, output file) |
| `labelling/` | Claim sample for human validation of the LLM judge, failure-analysis sample |
| `logs/` | Logs of every run, including failed ones |
| `refs/` | Literature review and the citation verification record (`VERIFIED_REFS.md`) |
| `tests/` | pytest suite |
| `PROJECT_STATE.md` | Running project log (phase gates, decisions, deviations) |

## Environment

- Python 3.12 with [uv](https://docs.astral.sh/uv/); exact versions in `uv.lock` (`uv sync`).
- Developed in WSL2 Ubuntu 24.04 with one NVIDIA GPU. Generators are served by vLLM 0.33.0 from Docker
  (`scripts/serve_vllm_docker.sh`) at `http://127.0.0.1:8000/v1`.
- `source scripts/wsl_env.sh` sets the venv, Hugging Face cache and data root (`DATA_ROOT`); paths are for our
  machine and may need editing.
- Models are downloaded with `scripts/download_model.sh`; the retrieval corpus, indexes and model weights are not in
  the repository.

## Reproducing

1. `bash scripts/setup_env.sh` (environment), then `bash scripts/run_tests.sh`.
2. Build the BM25 index (`scripts/build_bm25.py`) over the DPR Wikipedia split.
3. Main runs: `bash scripts/phase5_all.sh`; additional generators, ablations and baselines: `bash scripts/phase6_runs.sh`.
4. Analysis: `scripts/confirmatory.py`, `scripts/exploratory_tests.py`, `scripts/figures.py --split test`.

Runs are long (GPU-hours); the drivers checkpoint and resume. The outputs we obtained are already in `results/`.

## Data and licences

The code in this repository is released under the MIT License (see `LICENSE`). The retrieval corpus is the DPR Wikipedia split (CC-BY-NC-4.0, research use). The datasets (biographies from
FActScore, ASQA, Natural Questions, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA) and the models are used under
their own licences. The splits in `data/splits/` are subsets of those datasets.

## AI assistance

The code, experiment runs and analysis were produced with an AI coding assistant (Claude, Anthropic, via Claude
Code) under the authors' direction; see `PROJECT_STATE.md` for the step-by-step record.
