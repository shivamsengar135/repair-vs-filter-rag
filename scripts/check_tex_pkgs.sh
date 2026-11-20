#!/usr/bin/env bash
# Diagnostic: are algorithm/algpseudocode/tikz in the Tectonic bundle (all are in TeX Live, i.e. on Overleaf)?
for f in algorithm.sty algpseudocode.sty algorithmicx.sty tikz.sty; do
  ~/.local/bin/tectonic -X bundle cat "$f" > /dev/null 2>&1 && echo "have $f" || echo "MISSING $f"
done
