# Project: Hallucination mitigation in LLMs (RAG + verification) — Q1 journal paper

**At the start of every session: read `PROJECT_STATE.md` first and continue from there.**

## Integrity rules (override everything else)
1. Never fabricate numbers, results, datasets, model names, APIs, papers, authors, DOIs or quotes. Unknown → write "UNKNOWN — need to verify" and verify from a primary source.
2. Every paper number traces to a results file produced by code in this repo; map it in `results/RESULTS_LEDGER.md` (script, config, seed, git commit, output file).
3. Every citation is checked against a real source (DOI / arXiv / ACL Anthology / venue page) before entering `references.bib`; record the check in `refs/VERIFIED_REFS.md`. Never cite from memory.
4. Find out what is true, not what the user hopes. Report negative/mixed results. No test-set tuning, no best-seed picking.
5. Baselines run in OUR setup (same retriever, corpus, generator, prompt budget, eval script). Numbers copied from papers go in a separate "reported in original paper (different setup)" table.
6. ≥3 seeds when non-deterministic; mean ± std; paired bootstrap (≥1000) or McNemar for headline comparisons; state α. Say "better" only when significant.
7. Show evidence (command, log tail, output path) before calling a step finished.
8. Unsure or blocked → stop and ask.

## Phases and gates
0 env → 1 literature → 2 method proposal → 3 preregistered plan → 4 implementation → 5 experiments → 6 analysis → 7 paper.
At every ★GATE write a gate summary (PROJECT_STATE.md + chat), then continue to the next phase without waiting
(user instruction 2026-11-16: "Next time continue the next phases by yourself."). Real blockers only the user can
clear (human labels, sudo, licences) are reported and worked around, never faked.

## Environment conventions
- Host is Windows 11; Python code runs in **WSL2 Ubuntu 24.04** (GPU visible there).
- **vLLM is served from Docker** (`vllm/vllm-openai:v0.33.0`, same version as the pinned Python package) via `scripts/serve_vllm_docker.sh`; client code in WSL talks to `http://127.0.0.1:8000/v1`. Native WSL vLLM does not start (no C compiler in WSL as of 2026-11-12).
- Model downloads: `scripts/download_model.sh` (plain HTTPS with retries; Xet fails on this network).
- Repo lives on the Windows drive: `/mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1`. Use **WSL git** (Windows has no git).
- `source scripts/wsl_env.sh` before running anything: sets `UV_PROJECT_ENVIRONMENT=~/venvs/halluc`, `HF_HOME`, `DATA_ROOT=~/halluc_data` (all on WSL ext4 — never put venvs, models or indexes under `/mnt/c`).
- Python env: `uv` + `pyproject.toml` + committed `uv.lock`. Add deps with `uv add`, never ad-hoc `pip install`.
- From the Claude Code PowerShell tool, run WSL commands as `wsl -d Ubuntu -- bash <script>`; put anything with non-trivial quoting in a script file under `scripts/` (inline quoting through PowerShell→WSL breaks).
- `sudo` in WSL needs the user's password — ask the user to run sudo commands themselves.
- Long jobs: tmux session inside WSL, log to `logs/`, checkpoint/resume. Never block on a multi-hour foreground job.
- Paper PDF locally: `bash scripts/compile_paper.sh` (Tectonic 0.15.0 in `~/.local/bin`, installed 2026-11-19 with the
  user's OK by `scripts/install_tectonic.sh`) compiles `overleaf/` to `paper_preview.pdf`; check layout there before sending a zip.
  Overleaf places *floating* figures differently from Tectonic (it held them all until after Sec. 6), so figures are
  non-floating `fixedfigure`s (defined in `paper/main.tex`); after any text change re-check for page-bottom gaps.
- Dev work uses a FIXED seeded subset saved to disk; full test sets only for final runs.
- Commit often. Log every run, including failed ones.

## Layout
- `scripts/` entry points and setup; `configs/` YAML; `src/` library code; `tests/` pytest
- `logs/` job logs; `results/` outputs + `RESULTS_LEDGER.md`; `refs/` literature + `VERIFIED_REFS.md`
