"""Print answers of the most recent run matching a glob. Usage: show_answers.py 'results/dev/bio/flare__*' [n] [chars] [fields...]"""
import glob
import sys

from hallu.store import read_jsonl

run = sorted(glob.glob(sys.argv[1]))[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
w = int(sys.argv[3]) if len(sys.argv) > 3 else 400
fields = sys.argv[4:]
print("run:", run)
for r in read_jsonl(f"{run}/items.jsonl")[:n]:
    print(" ".join(str(r.get(f)) for f in fields), "|", r["question"][:60], "|", r["answer"][:w].replace("\n", " "))
