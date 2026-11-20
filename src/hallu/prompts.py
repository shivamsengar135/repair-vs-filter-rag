"""Loading and filling of the fixed prompt templates in configs/prompts/."""
from __future__ import annotations

import string
from pathlib import Path

import yaml


class Prompts:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = data.pop("version")
        self.templates: dict[str, str] = {k: v.rstrip("\n") for k, v in data.items()}

    def fields(self, name: str) -> set[str]:
        return {f for _, f, _, _ in string.Formatter().parse(self.templates[name]) if f}

    def fill(self, name: str, **kw: str) -> str:
        """Fill a template; refuses missing or unexpected fields so a typo cannot silently change a prompt."""
        need = self.fields(name)
        if set(kw) != need:
            raise KeyError(f"prompt {name!r} needs fields {sorted(need)}, got {sorted(kw)}")
        return self.templates[name].format(**kw)
