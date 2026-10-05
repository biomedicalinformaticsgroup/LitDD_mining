#!/usr/bin/env python3
"""Characterise the recall misses of the released map against the external truth sets.

For every (g2p_id, pmid) in the truth set that the released map does not recover, the miss
is categorised with ``measure_recall.classify_miss``:

  litdd_bert_negative : the screen classified the PMID negative, so it never reached the gate
  mapped_other        : the pipeline mapped the PMID to a different G2P entry
  llm_no_match        : the adjudicator returned NO MATCH
  not_in_final_map    : mapped to the entry but absent from the released map

Each miss is joined to NCBI publication types (from ``fetch_pmid_meta.py``) and tagged
``in_scope`` or ``out_of_scope_pubtype`` (review, editorial, comment, letter, meta-analysis
and similar). Writes ``deployed_misses.csv`` and ``miss_characterisation.csv`` to ``--out_dir``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from litdd.evaluation.external_recall import measure_recall as mr

OUT_OF_SCOPE_PUBTYPES = (
    "Review", "Editorial", "Comment", "News", "Retraction", "Published Erratum",
    "Biography", "Historical Article", "Guideline", "Practice Guideline",
    "Meta-Analysis", "Systematic Review", "Letter", "Congress", "Address",
)


def pub_scope(pubtypes) -> str:
    pt = str(pubtypes)
    return "out_of_scope_pubtype" if any(o in pt for o in OUT_OF_SCOPE_PUBTYPES) else "in_scope"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", required=True, help="truthsets.csv from build_truthsets.py")
    ap.add_argument("--litdd_map", required=True)
    ap.add_argument("--complete_df", required=True)
    ap.add_argument("--meta", nargs="+", default=[], help="esummary metadata CSV(s): pmid,year,pubtypes,title")
    ap.add_argument("--min_year", type=int, default=1981)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> int:
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

    rows = [(g, p, mr.classify_miss(g, p, state))
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

    print(f"Misses: {len(md)} pairs / {md['pmid'].nunique()} unique PMIDs (min_year {args.min_year})")
    print(summary.to_string(index=False))
    n_out = int((md["pubscope"] == "out_of_scope_pubtype").sum())
    print(f"\nout-of-scope publication type: {n_out} / {len(md)} "
          f"({100 * n_out / max(len(md), 1):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
