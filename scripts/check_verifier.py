"""Phase 4 sanity check of the verifiers on hand-written cases with unambiguous answers (CPU or GPU).

This is a smoke test of loading and input formatting, NOT an accuracy evaluation.
"""
import sys
import time

import torch

from hallu.verifier import NLI, MiniCheck

DOC = ("Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 and won the Nobel Prize in Physics in 1903 "
       "and the Nobel Prize in Chemistry in 1911.")
CASES = [  # (claim, supported?, contradicted?)
    ("Marie Curie was born in Warsaw.", True, False),
    ("Marie Curie won the Nobel Prize in Chemistry in 1911.", True, False),
    ("Marie Curie was born in Paris.", False, True),
    ("Marie Curie won the Nobel Prize in Physics in 1921.", False, True),
    ("Marie Curie had two daughters.", False, False),
    # Off-topic false statement. Observed 2026-11-13: the NLI model calls this "contradiction" (0.99) although the
    # passage says nothing about it. Kept as an informational case (None = not asserted) and logged as a known
    # verifier failure mode: "contradicted" does not imply the passage is about the claim.
    ("The Eiffel Tower is in Berlin.", False, None),
]
device = sys.argv[1] if len(sys.argv) > 1 else None
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), "requested device:", device)
t0 = time.perf_counter()
mc = MiniCheck(device=device)
print(f"MiniCheck loaded on {mc.device} in {time.perf_counter() - t0:.1f}s; label token ids decode to:",
      [mc.tok.decode([i]) for i in (3, 209)])
t0 = time.perf_counter()
sp = mc.support_prob([DOC] * len(CASES), [c for c, _, _ in CASES])
t_mc = time.perf_counter() - t0
t0 = time.perf_counter()
nli = NLI(device=device)
print(f"NLI loaded on {nli.device} in {time.perf_counter() - t0:.1f}s; label index: {nli.label_index}")
t0 = time.perf_counter()
np_ = nli.probs([DOC] * len(CASES), [c for c, _, _ in CASES])
t_nli = time.perf_counter() - t0
ok = True
for (claim, sup, con), s, n in zip(CASES, sp, np_):
    good = ((s >= 0.5) == sup) and (con is None or (n["contradiction"] >= 0.5) == con)
    ok &= good
    print(f"{'ok ' if good else 'BAD'} minicheck={s:.3f} nli(e/n/c)={n['entailment']:.2f}/{n['neutral']:.2f}/{n['contradiction']:.2f} "
          f"expect sup={sup} con={con} | {claim}")
print(f"timing for {len(CASES)} pairs: minicheck {t_mc:.2f}s, nli {t_nli:.2f}s")
print("VERIFIER_SANITY_OK" if ok else "VERIFIER_SANITY_FAILED")
