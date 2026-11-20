"""Phase 1: download paper full text (arXiv HTML, falling back to ar5iv) and convert to plain text.

Tables are kept as one line per row with ' | ' between cells so result rows can be grepped.
Output goes to refs/raw/fulltext/<key>.txt — gitignored (copyrighted text; local reading copy only).
Each file starts with a header naming the exact source URL.

Usage: python3 scripts/lit/fetch_fulltext.py KEY=ARXIV_ID [KEY=ARXIV_ID ...]
"""
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_meta import get  # noqa: E402

OUT = Path("refs/raw/fulltext")
BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "figcaption", "caption", "table", "br", "figure"}


class Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.in_math = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style", "nav", "annotation", "annotation-xml"):
            self.skip += 1
        elif tag == "math":
            self.in_math += 1
            alt = a.get("alttext")
            if alt:
                self.out.append(alt)
        elif tag == "tr":
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("## ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "annotation", "annotation-xml"):
            self.skip = max(0, self.skip - 1)
        elif tag == "math":
            self.in_math = max(0, self.in_math - 1)
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip and not self.in_math:
            self.out.append(data)


def to_text(html: str) -> str:
    p = Text()
    p.feed(html)
    lines = [" ".join(l.split()) for l in "".join(p.out).splitlines()]
    return "\n".join(l for l in lines if l)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:]:
        key, aid = spec.split("=")
        dst = OUT / f"{key}.txt"
        if dst.exists() and dst.stat().st_size > 20000:
            print(f"skip {key} (have {dst.stat().st_size} bytes)")
            continue
        done = False
        for url in (f"https://arxiv.org/html/{aid}", f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            try:
                html = get(url, tries=2).decode("utf-8", "replace")
                txt = to_text(html)
                if len(txt) < 15000 or "Conversion to HTML had a Fatal error" in txt:
                    print(f"  {key}: {url} too short / failed ({len(txt)} chars)")
                    continue
                dst.write_text(f"# SOURCE: {url}\n# arXiv:{aid}\n{txt}")
                print(f"ok   {key:<18} {len(txt):>7} chars  tables~{txt.count('Table ')}  {url}")
                done = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {key}: {url} error {str(e)[:100]}")
        if not done:
            print(f"FAIL {key} {aid}")


if __name__ == "__main__":
    main()
