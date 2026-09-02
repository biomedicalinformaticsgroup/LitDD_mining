#!/usr/bin/env python3
"""Amalgamate phenotype.hpoa HPO terms per G2P entry, for the LLM prompt.

Linkage (per the 2026-09-02 design): the G2P entry's ``disease MONDO`` is the primary
disease identity; entries without one borrow the MONDO their ``disease mim`` maps to via
the OMIM->MONDO pairs present in the same DDG2P export. Every HPOA disease (OMIM-keyed)
whose OMIM maps to that MONDO contributes its terms — i.e. terms are amalgamated per
disease identity, not per OMIM series entry — and an entry whose OMIM has no MONDO link
still uses its own OMIM row directly.

Each term carries its name (hp.obo), the frequency as given (an HP:0040xxx bucket is
rendered as its label, n/m and percentages verbatim), and the provenance PMIDs from the
``reference`` column so the prompt builder can drop terms sourced from the very abstract
being adjudicated (leakage guard). ``qualifier == NOT`` rows are excluded; aspects P
(phenotype) and I (inheritance) are kept.

    python litdd/evaluation/build_hpo_terms.py \\
        --hpoa revision/phenotype.hpoa --hp_obo revision/context_build/hp.obo \\
        --g2p_csv revision/G2P_DD_2026-06-24.csv \\
        --out_json revision/llm_eval/hpo_terms_2026.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import pandas as pd

FREQ_BUCKETS = {"HP:0040280": "obligate (100%)", "HP:0040281": "very frequent (80-99%)",
                "HP:0040282": "frequent (30-79%)", "HP:0040283": "occasional (5-29%)",
                "HP:0040284": "very rare (1-4%)", "HP:0040285": "excluded (0%)"}


def load_hp_names(path: str) -> dict[str, str]:
    names, hid = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                hid = None
            elif line.startswith("id: HP:"):
                hid = line[4:]
            elif line.startswith("name: ") and hid:
                names[hid] = line[6:]
                hid = None
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hpoa", required=True)
    ap.add_argument("--hp_obo", required=True)
    ap.add_argument("--g2p_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    names = load_hp_names(args.hp_obo)
    g = pd.read_csv(args.g2p_csv)
    g["omim"] = ("OMIM:" + g["disease mim"].astype(str).str.replace(r"\.0$", "", regex=True)
                 ).where(g["disease mim"].notna())
    g["mondo"] = g["disease MONDO"].astype(str).str.strip().where(g["disease MONDO"].notna())

    omim2mondo = {r.omim: r.mondo for r in g.dropna(subset=["omim", "mondo"]).itertuples()}
    mondo2omims = defaultdict(set)
    for o, m in omim2mondo.items():
        mondo2omims[m].add(o)

    h = pd.read_csv(args.hpoa, sep="\t", comment="#", low_memory=False)
    h = h[h["qualifier"].fillna("") != "NOT"]
    h = h[h["aspect"].isin(["P", "I"])]
    by_disease: dict[str, list] = defaultdict(list)
    for r in h.itertuples(index=False):
        by_disease[r.database_id].append(r)

    out, n_entries, n_terms = {}, 0, 0
    for r in g.itertuples():
        gid = g.loc[r.Index, "g2p id"]
        mondo = r.mondo if isinstance(r.mondo, str) and r.mondo.startswith("MONDO") else (
            omim2mondo.get(r.omim))
        omims = set(mondo2omims.get(mondo, set()))
        if isinstance(r.omim, str):
            omims.add(r.omim)
        terms: dict[str, dict] = {}
        for o in sorted(omims):
            for row in by_disease.get(o, []):
                t = terms.setdefault(row.hpo_id, {"id": row.hpo_id,
                                                  "name": names.get(row.hpo_id, row.hpo_id),
                                                  "freq": [], "pmids": []})
                fr = row.frequency if isinstance(row.frequency, str) else None
                if fr:
                    fr = FREQ_BUCKETS.get(fr, fr)
                    if fr not in t["freq"]:
                        t["freq"].append(fr)
                for m in re.findall(r"PMID:(\d+)", str(row.reference)):
                    if m not in t["pmids"]:
                        t["pmids"].append(m)
        if terms:
            out[str(gid)] = sorted(terms.values(), key=lambda t: t["id"])
            n_entries += 1
            n_terms += len(terms)
    with open(args.out_json, "w") as f:
        json.dump({"__provenance__": {"hpoa": args.hpoa, "hp_obo": args.hp_obo,
                                      "g2p_csv": args.g2p_csv,
                                      "entries_with_terms": n_entries,
                                      "mean_terms_per_entry": round(n_terms / max(1, n_entries), 1)},
                   **out}, f)
    print(f"{n_entries}/{len(g)} G2P entries with HPO terms; mean {n_terms / max(1, n_entries):.1f} "
          f"terms/entry; wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
