#!/usr/bin/env python3
"""Cross-encoder score-threshold sweep (Reviewer 3 R3.6): how the >=0.9 gate trades external
recall against corpus size (a proxy for false-positive volume at deployment scale).

For each cutoff t, the mined corpus is the set of (PMID -> G2P) mappings the LLM assigned whose
cross-encoder score >= t (gene-mention filter off, i.e. the `relaxed` variant). Reports external
recall (combined premined+HPOA+ClinGen truth, per disease) and the corpus size at each t, so the
0.9 choice can be read off the recall-vs-size curve rather than asserted. Reuses measure_recall.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import measure_recall as mr
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", default="revision/external_recall/truthsets.csv")
    ap.add_argument("--complete_df", required=True)
    ap.add_argument("--pmid_years", default=None)
    ap.add_argument("--min_year", type=int, default=1981)
    ap.add_argument("--cutoffs", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    ts = pd.read_csv(args.truthsets, dtype=str)
    state, years = mr.load_pipeline_state(args.complete_df)
    if args.pmid_years:
        ey = pd.read_csv(args.pmid_years, dtype=str)
        for p, y in zip(ey["pmid"], pd.to_numeric(ey["year"], errors="coerce")):
            if y == y and p not in years:
                years[p] = int(y)
    if args.min_year:
        ts = ts[ts["pmid"].map(lambda p: years.get(p, 9999) >= args.min_year)]

    truth: dict[str, set[str]] = defaultdict(set)
    for src, k, p in zip(ts["source"], ts["key"], ts["pmid"]):
        if src in mr.REPORTABLE:
            truth[k].add(p)

    rows = []
    for t in [float(x) for x in args.cutoffs.split(",")]:
        mined = mr.mined_relaxed(state, t)
        micro, macro, n_dis, n_truth = mr.recall_stats(truth, mined)
        n_map = sum(len(v) for v in mined.values())
        n_pmid = len(set().union(*mined.values())) if mined else 0
        rows.append({"cutoff": t, "micro_recall": round(micro, 3), "macro_recall": round(macro, 3),
                     "corpus_mappings": n_map, "corpus_pmids": n_pmid})

    out = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out_dir) / "threshold_sweep.csv", index=False)
    print("Cross-encoder threshold sweep (combined external truth, relaxed; R3.6):")
    print(out.to_string(index=False))
    base = out[out.cutoff == 0.9]
    if len(base):
        b = base.iloc[0]
        print(f"\nAt the deployed 0.9 gate: recall {b.micro_recall}/{b.macro_recall}, "
              f"corpus {int(b.corpus_mappings):,} mappings / {int(b.corpus_pmids):,} PMIDs.")


if __name__ == "__main__":
    main()
