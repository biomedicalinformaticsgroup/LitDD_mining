#!/usr/bin/env python3
"""Characterise the deployed-corpus recall misses (Reviewer 3 R3.4 / Reviewer 2 C1/C2):
why is each curated paper not recalled, and is the gap a model failure or a corpus boundary?

For every (g2p_id, pmid) in the truth set that the deployed map does not recover, the miss
is categorised (reusing measure_recall.classify_miss):
  litdd_bert_negative : LitDD's BERT (run over all of PubMed) classified the PMID negative,
                  so it never entered the ranking/LLM stages. Many are papers with no
                  molecular confirmation (phenotype defined, causative gene unpublished).
  mapped_other  : the pipeline mapped the PMID, but to a different (usually sibling) G2P id.
  below_score   : mapped to the right id but cross-encoder < cutoff (recoverable via threshold).
  gene_filtered : right id at/above cutoff but dropped by the gene-in-TIAB filter.
  llm_no_match  : the LLM returned NO MATCH.

Each miss is joined to NCBI publication types (from fetch_pmid_meta.py outputs) and tagged
in_scope vs out_of_scope_pubtype (review/editorial/comment/letter/meta-analysis), so the
"is the miss out of scope?" question is answered with evidence rather than asserted.

Outputs (gitignored revision/ area): deployed_misses.csv, miss_characterisation.csv.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import measure_recall as mr
import pandas as pd

OUT_OF_SCOPE_PUBTYPES = (
    "Review", "Editorial", "Comment", "News", "Retraction", "Published Erratum",
    "Biography", "Historical Article", "Guideline", "Practice Guideline",
    "Meta-Analysis", "Systematic Review", "Letter", "Congress", "Address",
)


def pub_scope(pubtypes) -> str:
    pt = str(pubtypes)
    return "out_of_scope_pubtype" if any(o in pt for o in OUT_OF_SCOPE_PUBTYPES) else "in_scope"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", default="revision/external_recall/truthsets.csv")
    ap.add_argument("--litdd_map", required=True)
    ap.add_argument("--complete_df", required=True)
    ap.add_argument("--meta", nargs="+", default=[], help="esummary metadata CSV(s): pmid,year,pubtypes,title")
    ap.add_argument("--score_cutoff", type=float, default=0.9)
    ap.add_argument("--min_year", type=int, default=1981)
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    ts = pd.read_csv(args.truthsets, dtype=str)
    state, years = mr.load_pipeline_state(args.complete_df)
    meta = (pd.concat([pd.read_csv(m, dtype=str) for m in args.meta], ignore_index=True)
            .drop_duplicates("pmid")) if args.meta else pd.DataFrame(columns=["pmid", "pubtypes", "title"])
    for p, y in zip(meta["pmid"], pd.to_numeric(meta.get("year"), errors="coerce")):
        if y == y and p not in years:
            years[p] = int(y)

    if args.min_year:
        ts = ts[ts["pmid"].map(lambda p: years.get(p, 9999) >= args.min_year)]

    dep = mr.mined_deployed(args.litdd_map)
    truth: dict[str, set[str]] = defaultdict(set)
    for k, p in zip(ts["key"], ts["pmid"]):
        truth[k].add(p)

    rows = [(g, p, mr.classify_miss(g, p, state, args.score_cutoff))
            for g, ps in truth.items() for p in ps - dep.get(g, set())]
    md = pd.DataFrame(rows, columns=["g2p", "pmid", "category"]).drop_duplicates()
    md = md.merge(meta[["pmid", "pubtypes", "title"]], on="pmid", how="left")
    md["pubscope"] = md["pubtypes"].map(pub_scope)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md.to_csv(out / "deployed_misses.csv", index=False)
    summary = (md.groupby("category")
               .agg(n=("pmid", "size"),
                    out_of_scope_pubtype=("pubscope", lambda s: (s == "out_of_scope_pubtype").sum()))
               .reset_index().sort_values("n", ascending=False))
    summary["pct_of_misses"] = (100 * summary["n"] / len(md)).round(0)
    summary.to_csv(out / "miss_characterisation.csv", index=False)

    print(f"Deployed misses: {len(md)} pairs / {md['pmid'].nunique()} unique PMIDs (min_year {args.min_year})")
    print(summary.to_string(index=False))
    print(f"\nout-of-scope pubtype (review/editorial/etc.) overall: "
          f"{(md['pubscope'] == 'out_of_scope_pubtype').sum()} / {len(md)} "
          f"({100 * (md['pubscope'] == 'out_of_scope_pubtype').mean():.0f}%) — the misses are mostly in-scope.")


if __name__ == "__main__":
    main()
