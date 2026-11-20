"""Markdown table of closed-book / standard-RAG short-form results for one generator on one split (ledger helper).
Usage: python scripts/qa_table.py --model-tag Mistral-7B-Instruct-v0.3 [--split test]"""
import argparse
import glob
import json
from pathlib import Path

SETS = [("popqa", "PopQA"), ("nq", "NQ"), ("triviaqa", "TriviaQA"), ("hotpotqa", "HotpotQA"), ("2wiki", "2Wiki")]


def run(split: str, ds: str, method: str, tag: str) -> tuple[dict, str]:
    c = [Path(d) for d in glob.glob(f"results/{split}/{ds}/{method}__{tag}__*") if "__rep" not in d
         and (Path(d) / "summary.json").exists()]
    if len(c) != 1:
        raise SystemExit(f"expected 1 run for {split}/{ds}/{method}/{tag}, found {len(c)}: {c}")
    s = json.loads((c[0] / "summary.json").read_text())
    return s.get("metrics", s), c[0].name.rsplit("__", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    print("| Set | closed-book match acc | closed-book EM | standard RAG match acc | RAG EM | RAG F1 | retrieval recall@5 "
          "| closed-book dir hash | RAG dir hash |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ds, name in SETS:
        cb, hcb = run(a.split, ds, "closed_book", a.model_tag)
        rg, hrg = run(a.split, ds, "rag", a.model_tag)
        print(f"| {name} | {cb['match_accuracy']:.3f} | {cb['em']:.3f} | {rg['match_accuracy']:.3f} | {rg['em']:.3f} "
              f"| {rg['f1']:.3f} | {rg['retrieval_recall_at_k']:.3f} | {hcb} | {hrg} |")


if __name__ == "__main__":
    main()
