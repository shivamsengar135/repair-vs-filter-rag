#!/usr/bin/env bash
# Fetch popqa_longtail_w_gs.jsonl from two independent third-party Hub copies and verify they agree.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
D="$DATA_ROOT/selfrag_eval"; mkdir -p "$D/a" "$D/b"
H="$UV_PROJECT_ENVIRONMENT/bin/hf"
HF_HUB_DISABLE_XET=1 "$H" download jesse-tong/selfrag-eval-data eval_data/popqa_longtail_w_gs.jsonl --repo-type dataset --local-dir "$D/a" | tail -n 1
HF_HUB_DISABLE_XET=1 "$H" download awinml/popqa_longtail_w_gs popqa_longtail_w_gs.jsonl --repo-type dataset --local-dir "$D/b" | tail -n 1
A="$D/a/eval_data/popqa_longtail_w_gs.jsonl"; B="$D/b/popqa_longtail_w_gs.jsonl"
sha256sum "$A" "$B"
cmp -s "$A" "$B" && echo "IDENTICAL" || echo "DIFFERENT"
echo "lines: $(wc -l < "$A")"
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$A" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
r = rows[0]
print("keys:", list(r.keys()))
print("ctxs per item:", sorted({len(x.get("ctxs", [])) for x in rows})[:6], "| first ctx keys:", list(r["ctxs"][0].keys()) if r.get("ctxs") else None)
print("example:", json.dumps({k: (v if k != "ctxs" else v[:1]) for k, v in r.items()})[:700])
EOF
