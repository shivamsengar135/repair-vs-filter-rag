"""Locate licence sections in a saved TII terms page and print the opening of the one matching a pattern.
Usage: read_license_html.py FILE [pattern]"""
import re
import sys
from html.parser import HTMLParser


class T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, a):
        self.skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style")

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


p = T()
p.feed(open(sys.argv[1], encoding="utf-8", errors="replace").read())
text = " ".join(" ".join(p.out).split())
heads = [(m.start(), m.group(0)) for m in re.finditer(r"(TII Falcon[- ]?LLM License[^.]{0,40}|Falcon[^.]{0,30}TII License Version [0-9.]+)", text)]
print("licence headings found:", [h for _, h in heads][:12])
pat = sys.argv[2] if len(sys.argv) > 2 else "2.0"
hit = next(((i, h) for i, h in heads if pat in h), None)
if hit:
    seg = text[hit[0]:hit[0] + 6000]
    for s in re.split(r"(?<=[.;])\s+", seg):
        if re.search(r"(research|commercial|Acceptable Use|royalt|grant|Falcon 3|attribution)", s, re.I) and 30 < len(s) < 500:
            print("-", s[:350])
else:
    print("no section matching", pat)
