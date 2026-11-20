"""Extract text from a local PDF into refs/raw/fulltext/<key>.txt (gitignored reading copy).
Run with an ephemeral pypdf:  uv run --no-project --with pypdf python -I scripts/lit/pdf_to_text.py PDF KEY
"""
import sys
from pathlib import Path

from pypdf import PdfReader

pdf, key = sys.argv[1], sys.argv[2]
r = PdfReader(pdf)
out = [f"# SOURCE: local PDF {pdf} ({len(r.pages)} pages)"]
for i, p in enumerate(r.pages, 1):
    out.append(f"## [page {i}]")
    out.extend(l.rstrip() for l in (p.extract_text() or "").splitlines() if l.strip())
dst = Path("refs/raw/fulltext") / f"{key}.txt"
dst.write_text("\n".join(out))
print(f"{len(r.pages)} pages, {sum(len(l) for l in out)} chars -> {dst}")
print("title metadata:", (r.metadata or {}).get("/Title"))
