#!/usr/bin/env bash
# Rebuild overleaf/ + compile, print only the warnings that matter for layout, copy the PDF to the viewer folder.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/rebuild_and_compile.sh > /tmp/compile.out 2>&1 || { tail -n 30 /tmp/compile.out; exit 1; }
grep -E "Overfull|Underfull|already defined|undefined refer|Error" /tmp/compile.out | head -n 20 || true
grep -E "COMPILED" /tmp/compile.out
cp paper_preview.pdf "${1:-/dev/null}" 2>/dev/null || true
