#!/usr/bin/env python3
"""Quantify the cascade funnel on the deployed corpus (Reviewer 2 R2-P3 / R2-C1).

Reports how many records survive each pipeline stage (BERT screen -> cross-encoder +
LLM mapping -> gene-mention/score gate), i.e. the attrition the reviewer asks us to
measure on the deployed corpus rather than the balanced test set. Optionally samples the
mappings *dropped* by the final gene/score gate into a blinded worksheet, so an annotator
can check whether that filter removed true positives (its precision/recall cost — also
relevant to the gene-in-TIAB attrition point R2-C1 / R3.4).

Inputs are the small published map files plus, optionally, a parquet/CSV of BERT-positive
rows for the first stage count and a tiab source for the dropped-set worksheet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _explode_llm(df: pd.DataFrame, id_col: str) -> set:
    """Set of (pmid, g2p_id) from a (pmid, llm_dis_map/g2p_id) frame."""
    pairs = set()
    for pmid, ans in zip(df["pmid"].astype(int), df[id_col].astype(str)):
        if ans.strip().upper() in ("", "NO MATCH", "NAN"):
            continue
        for gid in (x.strip() for x in ans.split(";")):
            if gid.upper().startswith("G2P"):
                pairs.add((pmid, gid))
    return pairs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm_map", required=True, help="CSV pmid,llm_dis_map (LLM output, pre gene/score gate)")
    ap.add_argument("--final_map", required=True, help="CSV pmid,g2p_id (deployed corpus, post gate)")
    ap.add_argument("--bert_positive", default=None, help="Parquet of BERT-positive rows (first-stage count)")
    ap.add_argument("--tiab_source", default=None,
                    help="Parquet/CSV with pmid+tiab to build the dropped-set worksheet")
    ap.add_argument("--out_dir", default="revision/precision_audit")
    ap.add_argument("--dropped_n", type=int, default=100, help="Dropped-set sample size")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    llm = pd.read_csv(args.llm_map)
    final = pd.read_csv(args.final_map)
    llm_col = "llm_dis_map" if "llm_dis_map" in llm.columns else llm.columns[1]
    final_col = "g2p_id" if "g2p_id" in final.columns else final.columns[1]

    llm_pairs = _explode_llm(llm, llm_col)
    final_pairs = _explode_llm(final, final_col)

    # Funnel of mapping counts.
    stages = []
    if args.bert_positive:
        import pyarrow.parquet as pq
        stages.append(("BERT-positive abstracts", pq.ParquetFile(args.bert_positive).metadata.num_rows))
    stages.append(("LLM-mapped (non-NO-MATCH) mappings", len(llm_pairs)))
    stages.append(("After gene/score gate (deployed)", len(final_pairs)))

    print("=== Cascade funnel (R2-P3) ===")
    prev = None
    rows = []
    for name, n in stages:
        retained = "" if prev is None else f"{100*n/prev:5.1f}% of previous"
        print(f"  {name:42s} {n:>9,}  {retained}")
        rows.append({"stage": name, "n": n,
                     "pct_of_previous": None if prev is None else round(100 * n / prev, 2)})
        prev = n
    pd.DataFrame(rows).to_csv(out / "cascade_funnel.csv", index=False)

    dropped = sorted(llm_pairs - final_pairs)
    print(f"\nMappings dropped by the gene/score gate: {len(dropped):,}")

    if dropped and args.tiab_source:
        tiab = (pd.read_csv(args.tiab_source) if str(args.tiab_source).endswith(".csv")
                else pd.read_parquet(args.tiab_source))
        tiab = tiab.rename(columns={c: c.lower() for c in tiab.columns})
        if "tiab" not in tiab.columns:
            tiab["tiab"] = (tiab.get("title", "").fillna("") + " " + tiab.get("abstract", "").fillna("")).str.strip()
        tiab = tiab[["pmid", "tiab"]].drop_duplicates("pmid")
        tiab["pmid"] = tiab["pmid"].astype(int)

        dd = pd.DataFrame(dropped, columns=["pmid", "assigned_g2p_id"])
        idx = rng.choice(len(dd), size=min(args.dropped_n, len(dd)), replace=False)
        samp = dd.iloc[idx].merge(tiab, on="pmid", how="left")
        samp.insert(0, "drop_id", [f"D{i:03d}" for i in range(len(samp))])
        samp["verdict"] = ""   # correct mapping wrongly dropped? correct | incorrect | uncertain
        samp["notes"] = ""
        samp.to_csv(out / "dropped_set_worksheet.csv", index=False)
        print(f"Wrote {len(samp)} dropped mappings to {out}/dropped_set_worksheet.csv "
              "(annotate to estimate the filter's true-positive cost).")
    elif dropped:
        print("(pass --tiab_source to build the dropped-set audit worksheet)")


if __name__ == "__main__":
    main()
