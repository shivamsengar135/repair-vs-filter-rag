"""Word count of the abstract in paper/main.tex (IP&M limit: 250 words)."""
import re
from pathlib import Path

tex = Path("paper/main.tex").read_text(encoding="utf-8")
body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
body = re.sub(r"\$[^$]*\$", "X", body).replace("\\%", "%")
print("abstract words:", len(body.split()))
