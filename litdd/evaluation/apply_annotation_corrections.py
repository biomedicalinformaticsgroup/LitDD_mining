#!/usr/bin/env python3
"""Apply reviewed label corrections to the annotation files, traceably.

``data/annotation_corrections.csv`` (pmid, g2p_id_from, g2p_id_to, reason, date, by) records
every clinician-reviewed relabelling. This script applies it to:

  --anno_csv       the full pair annotation (g2p_id, pmid, tiab, label) -> --anno_out
                   (positive rows for (pmid, g2p_id_from) are moved to g2p_id_to; an existing
                   row for (pmid, g2p_id_to) is merged with label = max)
  --pmid_csv       the repo's data/annotated_pmid.csv (pmid, g2p_lgmde, label) in place, with the
                   thread re-rendered for g2p_id_to from --g2p_csv

Every applied / unmatched correction is printed; unmatched ones abort.

    python litdd/evaluation/apply_annotation_corrections.py \\
        --corrections data/annotation_corrections.csv \\
        --anno_csv $REF/g2p_id_tiab_anno_df_FINAL.csv \\
        --anno_out revision/external_recall/g2p_id_tiab_anno_df_FINAL_corrected.csv \\
        --pmid_csv data/annotated_pmid.csv --g2p_csv $REF/G2P_DD_2025-02-15.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from litdd.threads import build_lgmde_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--anno_csv", required=True)
    ap.add_argument("--anno_out", required=True)
    ap.add_argument("--pmid_csv", default=None)
    ap.add_argument("--g2p_csv", default=None, help="export used to render threads in --pmid_csv")
    args = ap.parse_args()

    corr = pd.read_csv(args.corrections, dtype=str)
    anno = pd.read_csv(args.anno_csv, dtype={"pmid": str, "g2p_id": str})
    anno["pmid"] = anno["pmid"].astype(float).astype(int).astype(str)
    applied = 0
    for c in corr.itertuples(index=False):
        m = (anno["pmid"] == c.pmid) & (anno["g2p_id"] == c.g2p_id_from) & (anno["label"] == 1)
        if not m.any():
            raise SystemExit(f"[ERROR] no positive row for pmid {c.pmid} / {c.g2p_id_from}")
        anno.loc[m, "g2p_id"] = c.g2p_id_to
        applied += int(m.sum())
    before = len(anno)
    anno = (anno.groupby(["pmid", "g2p_id"], as_index=False)
            .agg(tiab=("tiab", "first"), label=("label", "max")))
    anno = anno[["g2p_id", "pmid", "tiab", "label"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.anno_out)), exist_ok=True)
    anno.to_csv(args.anno_out, index=False)
    print(f"full annotation: {applied} positive rows relabelled, {before - len(anno)} merged; "
          f"wrote {args.anno_out} ({len(anno)} rows)")

    if args.pmid_csv:
        if not args.g2p_csv:
            raise SystemExit("--pmid_csv needs --g2p_csv to render the corrected thread")
        panel = build_lgmde_map(args.g2p_csv)
        rep = pd.read_csv(args.pmid_csv, dtype={"pmid": str})
        rep["_id"] = rep["g2p_lgmde"].str.split(" - ", n=1).str[0].str.strip()
        n = 0
        for c in corr.itertuples(index=False):
            m = (rep["pmid"] == c.pmid) & (rep["_id"] == c.g2p_id_from) & (rep["label"] == 1)
            if not m.any():
                print(f"[note] {args.pmid_csv}: no row for pmid {c.pmid} / {c.g2p_id_from} (skipped)")
                continue
            if c.g2p_id_to not in panel:
                raise SystemExit(f"[ERROR] {c.g2p_id_to} not in {args.g2p_csv}")
            rep.loc[m, "g2p_lgmde"] = panel[c.g2p_id_to]
            n += int(m.sum())
        rep = rep.drop(columns="_id").drop_duplicates(subset=["pmid", "g2p_lgmde", "label"])
        rep.to_csv(args.pmid_csv, index=False, lineterminator="\r\n")  # file is CRLF
        print(f"{args.pmid_csv}: {n} rows relabelled in place ({len(rep)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
