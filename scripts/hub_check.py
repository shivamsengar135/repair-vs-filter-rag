"""Phase 3: verify candidate models / datasets on the Hugging Face Hub and repos on GitHub (read-only API calls).

Nothing is downloaded. Candidate IDs are guesses until the API confirms them; a 404 is a valid, recorded outcome.
Usage: python3 scripts/hub_check.py SPEC.json OUT.json
SPEC.json: {"models": [...], "datasets": [...], "model_searches": [...], "dataset_searches": [...], "github": [...]}
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "research-env-check/0.1"}


def get(url: str):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return {"_http_error": e.code}
            time.sleep(4 * (i + 1))
        except Exception:  # noqa: BLE001
            time.sleep(4 * (i + 1))
    return {"_http_error": "unreachable"}


def model_row(mid: str) -> dict:
    d = get(f"https://huggingface.co/api/models/{mid}?blobs=true")
    if "_http_error" in d:
        return {"id": mid, "error": d["_http_error"]}
    card = d.get("cardData") or {}
    st = (d.get("safetensors") or {}).get("total")
    size = sum((s.get("size") or 0) for s in d.get("siblings", []) if s["rfilename"].endswith((".safetensors", ".bin")))
    return {"id": d.get("id"), "gated": d.get("gated"), "license": card.get("license"), "license_name": card.get("license_name"),
            "params_B": round(st / 1e9, 2) if st else None, "weights_GB": round(size / 1e9, 1),
            "lastModified": (d.get("lastModified") or "")[:10], "downloads": d.get("downloads"), "sha": (d.get("sha") or "")[:12],
            "pipeline": d.get("pipeline_tag"), "library": d.get("library_name")}


def dataset_row(did: str) -> dict:
    d = get(f"https://huggingface.co/api/datasets/{did}")
    if "_http_error" in d:
        return {"id": did, "error": d["_http_error"]}
    card = d.get("cardData") or {}
    return {"id": d.get("id"), "gated": d.get("gated"), "license": card.get("license"),
            "lastModified": (d.get("lastModified") or "")[:10], "downloads": d.get("downloads"), "sha": (d.get("sha") or "")[:12],
            "configs": [c.get("config_name") for c in (card.get("configs") or [])][:12]}


def search(kind: str, q: str, n: int = 6) -> list[dict]:
    extra = "&filter=text-generation" if kind == "models" and not q.startswith("!") else ""
    q = q.lstrip("!")
    d = get(f"https://huggingface.co/api/{kind}?search={urllib.parse.quote(q)}&sort=downloads&direction=-1&limit={n}{extra}")
    if isinstance(d, dict):
        return [{"error": d.get("_http_error")}]
    return [{"id": x.get("id"), "downloads": x.get("downloads"), "gated": x.get("gated"), "created": (x.get("createdAt") or "")[:10]} for x in d]


def gh_row(repo: str) -> dict:
    d = get(f"https://api.github.com/repos/{repo}")
    if "_http_error" in d:
        return {"repo": repo, "error": d["_http_error"]}
    return {"repo": d.get("full_name"), "license": (d.get("license") or {}).get("spdx_id"), "pushed": (d.get("pushed_at") or "")[:10],
            "stars": d.get("stargazers_count"), "archived": d.get("archived")}


def main() -> None:
    spec = json.load(open(sys.argv[1]))
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": [], "datasets": [], "model_searches": {}, "dataset_searches": {}, "github": []}
    print("## MODELS")
    for m in spec.get("models", []):
        r = model_row(m); out["models"].append(r); print(json.dumps(r))
    print("## DATASETS")
    for d in spec.get("datasets", []):
        r = dataset_row(d); out["datasets"].append(r); print(json.dumps(r))
    print("## MODEL SEARCHES (top by downloads)")
    for q in spec.get("model_searches", []):
        r = search("models", q); out["model_searches"][q] = r
        print(q, "->", "; ".join(f"{x.get('id')} ({x.get('downloads')}, {x.get('created')}{', gated' if x.get('gated') else ''})" for x in r))
    print("## DATASET SEARCHES")
    for q in spec.get("dataset_searches", []):
        r = search("datasets", q); out["dataset_searches"][q] = r
        print(q, "->", "; ".join(f"{x.get('id')} ({x.get('downloads')})" for x in r))
    print("## GITHUB")
    for g in spec.get("github", []):
        r = gh_row(g); out["github"].append(r); print(json.dumps(r))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
