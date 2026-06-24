#!/usr/bin/env python3
"""Quantify the cascade funnel on the deployed corpus (Reviewer 2 R2-P3 / R2-C1).

Reports how many mappings survive each pipeline stage — BERT screen -> LLM mapping ->
cross-encoder score gate (>= --score_cutoff) -> gene-mention filter — i.e. the attrition
the reviewer asks us to measure on the deployed corpus. In particular it isolates the
**gene-in-TIAB filter's** attrition (R2-C1): mappings that pass the score gate but are
dropped because no linked gene is found in the abstract. It can also sample those dropped
mappings into a blinded worksheet so an annotator can estimate the filter's true-positive
cost (recall loss, R3.4).

Authoritative input is the complete pipeline parquet (``--complete_df``: one row per
BERT-positive abstract, with ``llm_dis_map`` and ``top5_cross``), which holds the *raw*
LLM mappings before any filter, plus the deployed corpus (``--final_map``: post gene/score
gate). NOTE: the small published map CSVs (e.g. pubmed_ddg2p_map.csv) are already filtered,
so they are NOT the raw LLM output — pass ``--complete_df`` for an accurate funnel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _g2p_ids(ans: str):
    ans = str(ans)
    if ans.strip().upper() in ("", "NO MATCH", "NAN"):
        return []
    return [x.strip() for x in ans.split(";") if x.strip().upper().startswith("G2P")]


def mapping_pairs(df: pd.DataFrame, score_cutoff: float):
    """From a complete df (pmid, llm_dis_map, top5_cross) return the set of all
    (pmid, g2p_id) LLM mappings and the subset whose assigned score >= score_cutoff."""
    all_pairs, score_pairs = set(), set()
    for pmid, ans, top5 in zip(df["pmid"].astype(int), df["llm_dis_map"], df["top5_cross"]):
        ids = _g2p_ids(ans)
        if not ids:
            continue
        scores = {}
        if top5 is not None:
            for c in top5:
                lab = c.get("label", "") if isinstance(c, dict) else ""
                gid = lab.split(" - ", 1)[0].strip()
                if gid:
                    scores[gid] = float(c.get("score", 0.0))
        for gid in ids:
            all_pairs.add((pmid, gid))
            if scores.get(gid, 0.0) >= score_cutoff:
                score_pairs.add((pmid, gid))
    return all_pairs, score_pairs


def _final_pairs(final_map: str):
    f = pd.read_csv(final_map)
    col = "g2p_id" if "g2p_id" in f.columns else f.columns[1]
    return {(int(p), g) for p, a in zip(f["pmid"], f[col]) for g in _g2p_ids(a)}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_df", required=True,
                    help="Complete pipeline parquet (pmid, llm_dis_map, top5_cross) — raw LLM mappings")
    ap.add_argument("--final_map", required=True, help="CSV pmid,g2p_id — deployed corpus (post gene/score gate)")
    ap.add_argument("--score_cutoff", type=float, default=0.9)
    ap.add_argument("--out_dir", default="revision/precision_audit")
    ap.add_argument("--dropped_n", type=int, default=100, help="Gene-dropped sample size for the worksheet")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.complete_df, columns=["pmid", "llm_dis_map", "top5_cross"])
    bert_n = len(df)
    all_pairs, score_pairs = mapping_pairs(df, args.score_cutoff)
    final_pairs = _final_pairs(args.final_map)

    stages = [
        ("BERT-positive abstracts", bert_n),
        ("LLM-mapped (non-NO-MATCH) mappings", len(all_pairs)),
        (f"Score gate (>= {args.score_cutoff}) mappings", len(score_pairs)),
        ("Gene-mention filter -> deployed corpus", len(final_pairs)),
    ]
    print("=== Cascade funnel (R2-P3 / R2-C1) ===")
    prev, rows = None, []
    for name, n in stages:
        retained = "" if prev is None else f"{100 * n / prev:5.1f}% of previous"
        print(f"  {name:42s} {n:>9,}  {retained}")
        rows.append({"stage": name, "n": n,
                     "pct_of_previous": None if prev is None else round(100 * n / prev, 2)})
        prev = n
    pd.DataFrame(rows).to_csv(out / "cascade_funnel.csv", index=False)

    # Gene-mention filter attrition: score-passing mappings dropped for lack of a gene mention.
    gene_dropped = score_pairs - final_pairs
    pct = 100 * len(gene_dropped) / len(score_pairs) if score_pairs else 0.0
    print(f"\nGene-mention filter attrition (R2-C1): {len(gene_dropped):,} score-passing "
          f"mappings dropped ({pct:.1f}% of score-passing).")

    # Blinded worksheet of gene-dropped mappings, to estimate the filter's true-positive cost.
    if gene_dropped:
        tiab = df.rename(columns={c: c.lower() for c in df.columns})[["pmid", "llm_dis_map"]]
        tiab = pd.read_parquet(args.complete_df, columns=["pmid", "tiab"]).drop_duplicates("pmid")
        tiab["pmid"] = tiab["pmid"].astype(int)
        dd = pd.DataFrame(sorted(gene_dropped), columns=["pmid", "assigned_g2p_id"])
        dd = dd.iloc[rng.choice(len(dd), size=min(args.dropped_n, len(dd)), replace=False)]
        samp = dd.merge(tiab, on="pmid", how="left")
        samp.insert(0, "drop_id", [f"D{i:03d}" for i in range(len(samp))])
        samp["verdict"] = ""   # would the gene filter wrongly drop a correct mapping? correct|incorrect|uncertain
        samp["notes"] = ""
        samp.to_csv(out / "gene_dropped_worksheet.csv", index=False)
        print(f"Wrote {len(samp)} gene-dropped mappings to {out}/gene_dropped_worksheet.csv "
              "(annotate to estimate the filter's recall cost, R3.4).")


if __name__ == "__main__":
    main()
