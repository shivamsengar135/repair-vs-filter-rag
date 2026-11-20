"""Read a Hugging Face model-API JSON response on stdin; print sha, licence, sizes. Usage: curl ... | python hf_model_info.py ID"""
import json
import sys

mid = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    print(f"{mid}: NO VALID RESPONSE")
    sys.exit()
if "error" in d:
    print(f"{mid}: ERROR {d['error']}")
    sys.exit()
lic = [t for t in d.get("tags", []) if t.startswith("license:")]
sib = d.get("siblings", [])
size = sum((s.get("size") or 0) for s in sib)
print(f"{mid}: sha {d.get('sha')} gated={d.get('gated')} {lic} files={len(sib)} total={size / 1e9:.2f} GB")
for s in sib:
    if s["rfilename"].endswith((".safetensors", ".bin", ".pt", ".onnx")):
        print(f"    {s['rfilename']} {(s.get('size') or 0) / 1e9:.3f} GB")
