"""Claim-vs-evidence verifiers (EXPERIMENT_PLAN.md §5.1).

MiniCheck (Flan-T5-Large) is used exactly as in the official implementation
(github.com/Liyan06/MiniCheck, minicheck/inference.py, read 2026-11-13):
    input  = "predict: " + document + <eos> + claim          (truncated to max_model_len tokens)
    output = softmax over the first decoder step's logits for token ids [3, 209]; index 1 = P(supported)
MiniCheck is binary (supported / not supported). Contradiction therefore comes from a separate 3-way NLI model.
"""
from __future__ import annotations

from typing import Sequence

import torch

MINICHECK_ID = "lytang/MiniCheck-Flan-T5-Large"
# Pinned to the authors' main revision. That revision ships only pytorch_model.bin; without use_safetensors=False
# transformers silently loads a bot-converted model.safetensors from an unmerged PR ref (observed 2026-11-13).
MINICHECK_REV = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
MINICHECK_LABEL_TOKEN_IDS = (3, 209)  # (not supported, supported) — from the official code
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_REV = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class MiniCheck:
    def __init__(self, device: str | None = None, max_model_len: int = 2048, batch_size: int = 32, half: bool = False):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = _device(device)
        self.tok = AutoTokenizer.from_pretrained(MINICHECK_ID, revision=MINICHECK_REV)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK_ID, revision=MINICHECK_REV, use_safetensors=False).to(self.device).eval()
        if half and self.device == "cuda":
            self.model = self.model.half()
        self.max_model_len, self.batch_size = max_model_len, batch_size

    @torch.no_grad()
    def support_prob(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        """P(claim is supported by doc) for aligned (doc, claim) pairs."""
        if len(docs) != len(claims):
            raise ValueError("docs and claims must be aligned")
        out: list[float] = []
        ids = torch.tensor(MINICHECK_LABEL_TOKEN_IDS)
        for i in range(0, len(docs), self.batch_size):
            texts = ["predict: " + self.tok.eos_token.join([d, c]) for d, c in zip(docs[i:i + self.batch_size], claims[i:i + self.batch_size])]
            enc = self.tok(texts, max_length=self.max_model_len, truncation=True, padding=True, return_tensors="pt").to(self.device)
            dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec).logits.squeeze(1)
            out.extend(torch.softmax(logits[:, ids.to(logits.device)].float(), dim=-1)[:, 1].cpu().tolist())
        return out


class NLI:
    """3-way NLI; returns P(entailment), P(neutral), P(contradiction) for (premise, hypothesis) pairs."""

    def __init__(self, device: str | None = None, max_len: int = 512, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = _device(device)
        self.tok = AutoTokenizer.from_pretrained(NLI_ID, revision=NLI_REV)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_ID, revision=NLI_REV).to(self.device).eval()
        # label order is read from the model config, never assumed
        self.label_index = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        assert set(self.label_index) == {"entailment", "neutral", "contradiction"}, self.label_index
        self.max_len, self.batch_size = max_len, batch_size

    @torch.no_grad()
    def probs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[dict]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be aligned")
        out: list[dict] = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(list(premises[i:i + self.batch_size]), list(hypotheses[i:i + self.batch_size]), max_length=self.max_len,
                           truncation="only_first", padding=True, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for row in p:
                out.append({k: float(row[j]) for k, j in self.label_index.items()})
        return out


ALIGNSCORE_ID = "liuyanyi/AlignScore-large-hf"  # HF-format conversion of AlignScore-large (base_model yzha/AlignScore)
ALIGNSCORE_REV = "cc13dc572fce8aa32778a437fb66a02d39a5b8e3"


class AlignScore:
    """AlignScore-large, scored as in the AlignScore code shipped with the converted checkpoint (`inferencer.py`,
    nlg_eval_mode "nli_sp", read 2026-11-17): pair = (premise, hypothesis) tokenised with truncation="only_first" to
    the tokenizer's max length; score = softmax(tri_layer(pooler_output))[:, 0] ("aligned"). Our premises are single
    passages and hypotheses single claims, so the original chunk/sentence max-mean aggregation reduces to one pair;
    across passages we take the max, as for MiniCheck. The model is RoBERTa-large + three linear heads; we build it
    from our own code and load the weights, so the repository's remote code is never executed."""

    def __init__(self, device: str | None = None, batch_size: int = 32):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

        self.device = _device(device)
        path = snapshot_download(ALIGNSCORE_ID, revision=ALIGNSCORE_REV)
        # concrete classes, not Auto*: the repo's config has an auto_map to remote code that must not be run
        self.tok = RobertaTokenizerFast.from_pretrained(path)
        cfg = RobertaConfig.from_pretrained(path)
        sd = load_file(f"{path}/model.safetensors")
        self.roberta = RobertaModel(cfg, add_pooling_layer=True)
        self.tri = torch.nn.Linear(cfg.hidden_size, 3)
        rob = {k[len("roberta."):]: v for k, v in sd.items() if k.startswith("roberta.")}
        missing, unexpected = self.roberta.load_state_dict(rob, strict=False)
        # position_ids buffers may be absent from checkpoints; every weight must load
        if [m for m in missing if not m.endswith("position_ids")] or unexpected:
            raise RuntimeError(f"AlignScore weights did not load cleanly: missing={missing} unexpected={unexpected}")
        self.tri.load_state_dict({"weight": sd["tri_layer.weight"], "bias": sd["tri_layer.bias"]})
        self.roberta.to(self.device).eval()
        self.tri.to(self.device).eval()
        self.batch_size = batch_size
        self.max_len = min(self.tok.model_max_length, 512)

    @torch.no_grad()
    def support_prob(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must be aligned")
        out: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            enc = self.tok(list(docs[i:i + self.batch_size]), list(claims[i:i + self.batch_size]), truncation="only_first",
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            pooled = self.roberta(**enc).pooler_output
            out.extend(torch.softmax(self.tri(pooled).float(), dim=-1)[:, 0].cpu().tolist())
        return out


class ScoreCache:
    """Disk cache of pair scores keyed by (scorer name, document, claim): scores are deterministic, so re-runs and
    ablations never recompute an identical pair. `fn(docs, claims) -> list` is called only for missing pairs."""

    def __init__(self, path, name: str):
        from hallu.store import JsonlStore
        self.store = JsonlStore(path, on_duplicate="skip")
        self.mem = {r["id"]: r["v"] for r in self.store}
        self.name = name
        self.hits = self.misses = 0

    def get(self, fn, docs: Sequence[str], claims: Sequence[str]) -> list:
        from hallu.store import text_key
        keys = [text_key(self.name, d, c) for d, c in zip(docs, claims)]
        todo = {k: (d, c) for k, d, c in zip(keys, docs, claims) if k not in self.mem}
        self.hits += len(keys) - len(todo)
        self.misses += len(todo)
        if todo:
            ks = list(todo)
            vals = fn([todo[k][0] for k in ks], [todo[k][1] for k in ks])
            for k, v in zip(ks, vals):
                self.mem[k] = v
                self.store.append({"id": k, "v": v})
        return [self.mem[k] for k in keys]


def aggregate_claim_scores(pair_scores: Sequence[Sequence[float]]) -> list[dict]:
    """Per claim, over its passages: support = max, best = argmax (0-based), or support 0 / best None if no passage."""
    res = []
    for scores in pair_scores:
        if len(scores) == 0:
            res.append({"support": 0.0, "best": None})
        else:
            b = max(range(len(scores)), key=lambda j: scores[j])
            res.append({"support": float(scores[b]), "best": b})
    return res
