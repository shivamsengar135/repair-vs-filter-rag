"""Phase 1: generate refs/VERIFIED_REFS.md from fetched metadata, so no field is typed from memory.

Sources: refs/raw/meta.json (arXiv API + Semantic Scholar), refs/raw/arxiv_search.json (arXiv API),
refs/raw/crossref.json (Crossref). Read depth is recorded by hand below and reflects what was
actually read in the session (F = main results table(s) read from the full text; A = abstract only).
"""
from __future__ import annotations

import json
from pathlib import Path

DATE = "2026-11-12"
FULL = {  # key -> what was read from the full text
    "selfrag": "setup (retriever, decoding), Table 2 (main), results discussion, ablation table",
    "crag": "Tables 1-6 (main, ablations, evaluator accuracy, overhead) with discussion, conclusion/limitation",
    "flare": "Tables 1-2, base-model statement",
    "cove": "Tables 1-3, base-model statement",
    "rarr": "abstract, Table 1 (partially: SQA and QReCC rows)",
    "selfcheckgpt": "dataset description, Table 2",
    "factscore": "Tables 1 and 3 (partially), estimator error-rate claim",
    "semantic_entropy": "abstract, metric definition (results tables not extracted)",
    "alce": "abstract, headline finding (results tables not extracted)",
    "safe_longfact": "abstract (partly), human-agreement and cost claims",
    "ragtruth": "abstract, Table 5",
    "adaptive_rag": "method section, dataset list, Table 1",
    "minicheck": "abstract (partly), Table 2 (partially: baseline rows)",
    "dragin": "baselines, implementation details, results discussion, Tables 2-3",
}
ABSTRACT = {"veriscore", "sep", "popqa", "r_tuning", "astute_rag", "fava", "clasheval", "seakr", "halueval",
            "chain_of_note", "instructrag",  # abstract (or most of it) read from the arXiv API record
            # read 2026-11-18 (Phase 7 prep) from the stored arXiv records:
            "rag2020", "summac", "survey_huang", "rag_survey_gao", "ret_robust", "ais", "attributed_qa", "alignscore"}
RECENT_FULL = {
    "2605.03534": "FULL READ: contributions, formulation, method, setup, Tables II/III/V/IX, limitations, conclusion",
    "2608.22228": "abstract, Table 2, failure analysis of NLI policy",
    "2510.11394": "FULL READ: intro, method stages, datasets/models, baselines, settings, main-results text, Tables 3-4, conclusion (Table 2 numbers not extracted)",
    "2501.12835": "FULL READ: contributions, setup, metrics, Table 1, sections 5-8, limitations",
    "2505.21072": "FULL READ: contributions, method, datasets/annotation, setup, results text, Tables 2 and 5, ablations, related work, limitations",
    "2603.27752": "FULL READ: contributions, datasets, baselines, Table 3, ablation text, reproducibility, cost, limitations",
    "2509.01455": "FULL READ (ar5iv): method evidence layer, training/inference, setup, results, ablations, limitations. No results tables exist; template placeholders present — low reliability",
}
RECENT = ["2605.03534", "2609.26229", "2603.10143", "2510.11394", "2604.18362", "2505.12662", "2502.12145",
          "2501.12835", "2505.04253", "2602.07213", "2502.01142", "2505.20871", "2604.20854", "2609.35794",
          "2610.35142", "2610.28340", "2608.22228", "2510.11483", "2505.21072", "2603.27752", "2509.21875",
          "2502.17125", "2512.20182", "2504.03579", "2508.14496", "2509.21557", "2609.14245", "2505.16973",
          "2502.18573", "2505.24830", "2601.21969", "2609.08267", "2505.04847", "2509.01455"]


# Phase 2 novelty-check papers (from refs/raw/arxiv_search_phase2.json) and what was read.
PHASE2 = {
    "2402.10978": "abstract; targeted passages (retention behaviour, limitations)",
    "2603.16817": "abstract; introduction and contributions; targeted passages on distractors and verifier cost",
    "2609.38222": "abstract; setup, single-hop reference result, limitations section",
    "2610.34872": "abstract only",
    "2506.20978": "abstract; introduction and conclusion passages",
    "2611.10335": "abstract only",
    "2604.13991": "abstract only",
    "2605.05244": "abstract only",
    "2604.10189": "abstract only",
    "2605.18792": "abstract; contributions, method overview, metrics, limitations appendix",
    "2609.22223": "abstract; introduction passages on cost and training",
    "2609.16564": "abstract only",
}


def esc(s: str | None) -> str:
    return (s or "").replace("|", "/").replace("\n", " ")


def main() -> None:
    meta = json.load(open("refs/raw/meta.json"))["records"]
    search = json.load(open("refs/raw/arxiv_search.json"))
    cross = json.load(open("refs/raw/crossref.json"))
    hits = {}
    for r in search["results"]:
        for h in r["hits"]:
            hits[h["id"].split("v")[0]] = h
    rows = []
    for r in meta:
        a, s = r["arxiv"], r["s2"]
        assert r["verified_title_match"], r["key"]
        first = a["authors"][0] + (" et al." if len(a["authors"]) > 1 else "")
        venue_author = a["journal_ref"] or a["comment"] or ""
        rows.append((r["key"], a["title"], f"{first} ({len(a['authors'])} authors)", a["published"][:4],
                     f"author-declared: {esc(venue_author)[:70] or '—'}; S2: {esc(s.get('venue')) or '—'} {s.get('year') or ''}",
                     f"arXiv:{a['arxiv_id_versioned']}", f"https://arxiv.org/abs/{r['arxiv_id']}",
                     FULL.get(r["key"], "abstract only" if r["key"] in ABSTRACT else "TITLE/METADATA ONLY — content not read")))
    for i in RECENT:
        h = hits[i]
        rows.append((f"arxiv{i.replace('.', '_')}", h["title"], f"{h['first_author']} (first author; full list in refs/raw/arxiv_search.json)",
                     h["published"][:4], "arXiv preprint; peer-review status NOT checked",
                     f"arXiv:{h['id']}", f"https://arxiv.org/abs/{i}", RECENT_FULL.get(i, "abstract only")))
    p2 = Path("refs/raw/arxiv_search_phase2.json")
    if p2.exists():
        hits2 = {h["id"].split("v")[0]: h for r in json.load(open(p2))["results"] for h in r["hits"]}
        for i, depth in PHASE2.items():
            h = hits2[i]
            rows.append((f"arxiv{i.replace('.', '_')}", h["title"],
                         f"{h['first_author']} (first author; full list in refs/raw/arxiv_search_phase2.json)",
                         h["published"][:4], "arXiv preprint; peer-review status NOT checked",
                         f"arXiv:{h['id']}", f"https://arxiv.org/abs/{i}", depth))
    for doi, c in cross.items():
        if "error" in c:
            continue
        rows.append((doi.split("/")[-1], c["title"], f"{c['authors'][0]} et al. ({len(c['authors'])} authors)",
                     str(c["issued"][0][0]), f"Crossref: {c['container']}" + (f", vol. {c['volume']}" if c["volume"] else "") + (f", pp. {c['page']}" if c["page"] else ""),
                     f"doi:{doi}", f"https://doi.org/{doi}",
                     "FULL READ: all 10 pages incl. appendix, from user-supplied PDF (text in refs/raw/fulltext/searag.txt); venue and authors on the PDF match Crossref"
                     if doi.endswith("3792598") else "metadata only (title/venue via Crossref); paper text NOT read"))
    # ---- Phase 7 (2026-11-18): models, datasets, tools and statistics cited for the experimental setup.
    # These are cited only for "we used X" / "X is defined in" statements, never for results; read = metadata only.
    p7 = Path("refs/raw/meta_phase7.json")
    if p7.exists():
        have = {row[0] for row in rows}
        for r in json.load(open(p7))["records"]:
            if r["key"] in have:  # already verified in Phase 1 (e.g. alignscore)
                continue
            a, s = r["arxiv"], r["s2"]
            assert r["verified_title_match"], r["key"]
            first = a["authors"][0] + (" et al." if len(a["authors"]) > 1 else "")
            rows.append((r["key"], a["title"], f"{first} ({len(a['authors'])} authors)", a["published"][:4],
                         f"author-declared: {esc(a['journal_ref'] or a['comment'] or '')[:70] or '—'}; S2: {esc(s.get('venue')) or '—'} {s.get('year') or ''}",
                         f"arXiv:{a['arxiv_id_versioned']}", f"https://arxiv.org/abs/{r['arxiv_id']}",
                         "metadata only (2026-11-18, arXiv API) — cited only as the source of a model/dataset/tool/method we used"))
    c7 = Path("refs/raw/crossref_phase7.json")
    keys7 = {"10.1162/tacl_a_00276": "naturalquestions", "10.1561/1500000019": "bm25_prf", "10.1007/BF02295996": "mcnemar1947",
             "10.1214/aos/1176344552": "efron1979", "10.1007/b106715": "vovk2005", "10.1017/pan.2023.20": "laurer2023"}
    if c7.exists():
        for doi, c in json.load(open(c7)).items():
            if "error" in c:
                continue
            authors = c["authors"] or ["Vladimir Vovk", "Alexander Gammerman", "Glenn Shafer"]  # vovk2005: Crossref has no authors; read from https://link.springer.com/book/10.1007/b106715 on 2026-11-18
            rows.append((keys7[doi], c["title"], f"{authors[0]} et al. ({len(authors)} authors)" if len(authors) > 1 else authors[0],
                         str(c["issued"][0][0]), f"Crossref: {c['container'] or c.get('publisher') or 'Springer (book)'}" + (f", vol. {c['volume']}" if c["volume"] else "") + (f", pp. {c['page']}" if c["page"] else ""),
                         f"doi:{doi}", f"https://doi.org/{doi}",
                         "metadata only (2026-11-18, Crossref" + ("; authors from the Springer book page" if not c["authors"] else "") + ") — cited only for a method/model we used"))
    rows.append(("holm1979", "A Simple Sequentially Rejective Multiple Test Procedure", "Sture Holm", "1979",
                 "Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65-70 (read on the JSTOR page; no DOI in Crossref)",
                 "JSTOR:4615733", "https://www.jstor.org/stable/4615733", "abstract read on JSTOR 2026-11-18 — cited for the Holm procedure"))
    rows.append(("falcon3", "The Falcon 3 family of Open Models", "TII Team", "2024",
                 "no paper: citation requested by the model card (@misc, December 2024)", "HF model card",
                 "https://huggingface.co/tiiuae/Falcon3-7B-Instruct", "model card read 2026-11-18 — cited as the model we used"))
    out = [
        "# Verified references", "",
        "Generated by `scripts/lit/build_verified_refs.py` from API responses saved under `refs/raw/` — do not edit by hand.",
        f"All checks performed {DATE}. A citation enters `references.bib` only if it has a row here.", "",
        "**What 'verified' means here**", "",
        "- Title, authors and submission date were read from the **arXiv API** record for the listed ID (primary source), or from **Crossref** for DOI-only entries.",
        "- For the 38 pre-2025 candidates the ID was proposed from memory and accepted only because the returned title matched expected keywords (`logs/phase1_fetch_meta.log`: 38/38).",
        "- **Venue is NOT yet verified against the venue's own page.** The venue column shows the authors' own arXiv comment and the Semantic Scholar aggregator value. Before a BibTeX entry is written, each venue must be confirmed on ACL Anthology / OpenReview / the publisher page.",
        "- **Phase 7 (2026-11-18): venues of cited papers were confirmed on the venue's own site** — ACL Anthology volume pages + official .bib, Crossref DOI records, the NeurIPS proceedings index, or OpenReview-hosted venues (DBLP-mirrored OpenReview records, CoRR, ARR submissions and withdrawn submissions do NOT count). Results: `refs/raw/venues.json`, `logs/phase7_venues.log`; scripts `scripts/lit/venue_check.py`, `scripts/lit/acl_volume_check.py`. Papers without a confirmed venue are cited as arXiv preprints. `paper/references.bib` is generated by `scripts/lit/build_bib.py` from these records only.",
        "- 'Read' states exactly what was read this session. 'abstract only' entries must not be cited for specific numbers; 'TITLE/METADATA ONLY' entries must not be cited for any content claim until read.", "",
        "| Key | Title | Authors | Year (arXiv v1) | Venue (unconfirmed, see above) | Identifier | Source URL | Read |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    Path("refs/VERIFIED_REFS.md").write_text("\n".join(out) + "\n")
    print(f"wrote refs/VERIFIED_REFS.md with {len(rows)} rows")


if __name__ == "__main__":
    main()
