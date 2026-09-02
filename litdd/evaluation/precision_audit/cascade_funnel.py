#!/usr/bin/env python3
"""Quantify the cascade funnel on the deployed corpus (Reviewer 2 R2-P3 / R2-C1).

**Two stage orders, because the pipeline was re-architected.** Select with ``--order``:

``legacy`` (the originally deployed cascade, and the order the published figures describe)::

    BERT screen -> LLM mapping -> score gate (>= cutoff) -> gene-mention filter

``gene_first`` (the revised cascade)::

    BERT screen -> gene gate -> cross-encoder (candidates only) -> LLM -> score gate

The gene-mention check moved ahead of the cross-encoder, which changes what its attrition
*means*: in ``legacy`` it is a final cleanup applied to already-scored mappings, so the 56.2%
it removed was the headline R2-C1 number; in ``gene_first`` it is a gate on abstracts, and the
quantity of interest is how many abstracts it admits and how many candidates each carries.
Reporting the same number under both labels would be misleading, so the stage list differs.

The gate's recall was measured before adoption (``litdd/evaluation/gene_filter_recall.py``):
98.8% of true pairs on the external curated sets, 99.3% with the HGNC name complement.

Under ``legacy`` this also isolates the gene filter's attrition (R2-C1) and can sample the
dropped mappings into a blinded worksheet so an annotator can estimate its true-positive cost
(recall loss, R3.4).

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
    ap.add_argument("--order", choices=["legacy", "gene_first"], default="legacy",
                    help="Cascade stage order. 'legacy' = the originally deployed pipeline "
                         "(gene filter last); 'gene_first' = the revised one (gene gate "
                         "before the cross-encoder).")
    ap.add_argument("--candidates_parquet", default=None,
                    help="gene_first only: output of gene_candidates.py, for the gate's "
                         "admitted-abstract and candidates-per-abstract counts.")
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
    # Rows != abstracts. PubMed updatefiles reissue records already present in the annual
    # baseline, so the same PMID can occupy several rows (51,613 of 782,230 in the 2025 run).
    # Report the unique-abstract count as the funnel head and the row count alongside it, or
    # the first stage silently overstates the corpus by the duplication rate (R2-C3 units).
    bert_rows = len(df)
    bert_n = df["pmid"].nunique()
    if bert_rows != bert_n:
        print(f"[note] {bert_rows - bert_n:,} duplicate PMID row(s) in the complete df "
              f"({bert_rows:,} rows -> {bert_n:,} unique abstracts)")
    all_pairs, score_pairs = mapping_pairs(df, args.score_cutoff)
    final_pairs = _final_pairs(args.final_map)

    if args.order == "gene_first":
        if not args.candidates_parquet:
            raise SystemExit("--order gene_first requires --candidates_parquet")
        cand = pd.read_parquet(args.candidates_parquet,
                               columns=["pmid", "candidate_g2p_ids"])
        n_admitted = cand["pmid"].nunique()
        n_pairs = int(cand["candidate_g2p_ids"].map(len).sum())
        stages = [
            ("BERT-positive abstracts (unique PMIDs)", bert_n, "abstracts"),
            ("Gene gate -> abstracts with >=1 candidate", n_admitted, "abstracts"),
            ("Cross-encoder (tiab, candidate) pairs scored", n_pairs, "pairs"),
            ("LLM-mapped (non-NO-MATCH) mappings", len(all_pairs), "mappings"),
            (f"Score gate (>= {args.score_cutoff}) -> deployed corpus",
             len(score_pairs), "mappings"),
        ]
    else:
        stages = [
            ("BERT-positive abstracts (unique PMIDs)", bert_n, "abstracts"),
            ("LLM-mapped (non-NO-MATCH) mappings", len(all_pairs), "mappings"),
            (f"Score gate (>= {args.score_cutoff}) mappings", len(score_pairs), "mappings"),
            ("Gene-mention filter -> deployed corpus", len(final_pairs), "mappings"),
        ]
    print(f"=== Cascade funnel — {args.order} order (R2-P3 / R2-C1) ===")
    prev, prev_unit, rows = None, None, []
    for name, n, unit in stages:
        # A percentage is only meaningful between stages measured in the same unit: the
        # gene-first cascade switches from abstracts to (abstract, candidate) pairs, and
        # "3.5x of previous" there would read as growth rather than a change of denominator.
        comparable = prev is not None and unit == prev_unit
        retained = f"{100 * n / prev:5.1f}% of previous" if comparable else f"({unit})"
        print(f"  {name:46s} {n:>12,}  {retained}")
        rows.append({"stage": name, "n": n, "unit": unit,
                     "pct_of_previous": round(100 * n / prev, 2) if comparable else None})
        prev, prev_unit = n, unit
    pd.DataFrame(rows).to_csv(out / "cascade_funnel.csv", index=False)

    if args.order == "gene_first":
        print("\n[note] The gene-dropped worksheet is a legacy-order artefact: with the gate "
              "ahead of the cross-encoder there is no set of 'score-passing mappings dropped "
              "for lack of a gene'. The gate's cost is measured directly instead — see "
              "litdd/evaluation/gene_filter_recall.py.")
        return 0

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
