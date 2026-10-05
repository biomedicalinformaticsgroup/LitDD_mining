#!/usr/bin/env python3
"""Count the abstracts, candidate pairs and mappings surviving each stage of the pipeline.

Stages, in pipeline order:

    screen-positive abstracts -> abstracts with a gate candidate -> (abstract, candidate)
    pairs offered to the adjudicator -> adjudicated mappings -> mappings in the released map

Inputs: the complete adjudication parquet (one row per screen-positive abstract with
``llm_dis_map``), the gate output (``candidates.parquet`` with ``pmid`` and
``candidate_g2p_ids``) and the released map CSV (``pmid``, ``g2p_id``). Writes
``cascade_funnel.csv`` to ``--out_dir`` and prints the table. A percentage of the previous
stage is reported only where both stages count the same unit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from litdd.evaluation.common import g2p_ids_split


def mapping_pairs(df: pd.DataFrame) -> set[tuple[int, str]]:
    """Every (pmid, g2p_id) pair in the adjudication output."""
    pairs = set()
    for pmid, ans in zip(df["pmid"].astype(int), df["llm_dis_map"]):
        for gid in g2p_ids_split(ans):
            if gid.upper().startswith("G2P"):
                pairs.add((pmid, gid))
    return pairs


def final_pairs(final_map: str) -> set[tuple[int, str]]:
    f = pd.read_csv(final_map)
    col = "g2p_id" if "g2p_id" in f.columns else f.columns[1]
    return {(int(p), g) for p, a in zip(f["pmid"], f[col]) for g in g2p_ids_split(a)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_df", required=True,
                    help="adjudication parquet with pmid and llm_dis_map for every screen-positive abstract")
    ap.add_argument("--candidates_parquet", required=True,
                    help="gene gate output with pmid and candidate_g2p_ids")
    ap.add_argument("--final_map", required=True, help="released map CSV (pmid, g2p_id)")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.complete_df, columns=["pmid", "llm_dis_map"])
    bert_rows = len(df)
    bert_n = df["pmid"].nunique()
    if bert_rows != bert_n:
        print(f"{bert_rows - bert_n:,} duplicate PMID rows ({bert_rows:,} rows, {bert_n:,} unique abstracts)")
    cand = pd.read_parquet(args.candidates_parquet, columns=["pmid", "candidate_g2p_ids"])
    stages = [
        ("Screen-positive abstracts (unique PMIDs)", bert_n, "abstracts"),
        ("Gene gate: abstracts with >=1 candidate", cand["pmid"].nunique(), "abstracts"),
        ("(abstract, candidate) pairs offered", int(cand["candidate_g2p_ids"].map(len).sum()), "pairs"),
        ("Adjudicated mappings (not NO MATCH)", len(mapping_pairs(df)), "mappings"),
        ("Mappings in the released map", len(final_pairs(args.final_map)), "mappings"),
    ]
    print("=== Cascade funnel ===")
    prev, prev_unit, rows = None, None, []
    for name, n, unit in stages:
        comparable = prev is not None and unit == prev_unit
        retained = f"{100 * n / prev:5.1f}% of previous" if comparable else f"({unit})"
        print(f"  {name:46s} {n:>12,}  {retained}")
        rows.append({"stage": name, "n": n, "unit": unit,
                     "pct_of_previous": round(100 * n / prev, 2) if comparable else None})
        prev, prev_unit = n, unit
    pd.DataFrame(rows).to_csv(out / "cascade_funnel.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
