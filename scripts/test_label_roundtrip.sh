#!/usr/bin/env bash
# Test: a CSV in the label tool's export format (CRLF, all fields quoted, BOM as Excel adds) is accepted by
# score_labelling.py. Works on a temp copy with dummy labels; the real labelling/ folder is not touched and the
# resulting kappa is meaningless (dummy labels), so it is not printed.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
cp labelling/.key_DO_NOT_OPEN.json "$T/"
python3 - "$T" <<'EOF'
import csv, sys, io
from pathlib import Path
rows = list(csv.DictReader(open("labelling/claims_to_label.csv", encoding="utf-8")))
buf = io.StringIO()
w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
w.writerow(["row", "topic", "claim", "evidence", "label", "note"])
for k, r in enumerate(rows):
    w.writerow([r["row"], r["topic"], r["claim"], r["evidence"], ["S", "NS", "U"][k % 3], ""])
Path(sys.argv[1], "claims_to_label.csv").write_text("﻿" + buf.getvalue(), encoding="utf-8")
EOF
bash scripts/py.sh scripts/score_labelling.py "$T/claims_to_label.csv" | python3 -c "import json,sys; d=json.load(sys.stdin); print('parsed OK:', d['n_labelled'], 'rows,', d['n_unclear'], 'U,', d['n_used'], 'used')"
rm -rf "$T"
