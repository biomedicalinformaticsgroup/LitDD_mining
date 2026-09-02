#!/usr/bin/env python3
"""Apply reviewed label corrections to the annotation files, traceably.

``data/annotation_corrections.csv`` (pmid, g2p_id_from, g2p_id_to, reason, date, by[, label])
records every clinician-reviewed change: a row with ``g2p_id_from`` relabels that positive pair
to ``g2p_id_to``; a row with an empty ``g2p_id_from`` ADDS a reviewed pair (``label`` 1 = new
positive, 0 = new negative) -- the 2026-09-01 review of the adjudicator's unlabelled extra
entries on co-reporting abstracts. This script applies it to:

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
    if "label" not in corr.columns:
        corr["label"] = "1"
    corr["g2p_id_from"] = corr["g2p_id_from"].fillna("")
    applied = added = 0
    tiab_of = anno.drop_duplicates("pmid").set_index("pmid")["tiab"]
    removed = 0
    for c in corr.itertuples(index=False):
        if str(c.label) == "remove":   # drop every pair for this entry (pmid '*' = all pmids)
            m = (anno["g2p_id"] == c.g2p_id_from) & ((c.pmid == "*") | (anno["pmid"] == c.pmid))
            removed += int(m.sum())
            anno = anno[~m]
            continue
        if c.g2p_id_from:      # relabel an existing positive pair
            m = (anno["pmid"] == c.pmid) & (anno["g2p_id"] == c.g2p_id_from) & (anno["label"] == 1)
            if not m.any():
                raise SystemExit(f"[ERROR] no positive row for pmid {c.pmid} / {c.g2p_id_from}")
            anno.loc[m, "g2p_id"] = c.g2p_id_to
            applied += int(m.sum())
        else:                  # add a reviewed pair (label 1 = new positive, 0 = new negative)
            if c.pmid not in tiab_of.index:
                raise SystemExit(f"[ERROR] pmid {c.pmid} not in the annotation (cannot add a pair)")
            anno = pd.concat([anno, pd.DataFrame([{"g2p_id": c.g2p_id_to, "pmid": c.pmid,
                                                    "tiab": tiab_of[c.pmid], "label": int(c.label)}])],
                             ignore_index=True)
            added += 1
    before = len(anno)
    anno = (anno.groupby(["pmid", "g2p_id"], as_index=False)
            .agg(tiab=("tiab", "first"), label=("label", "max")))
    anno = anno[["g2p_id", "pmid", "tiab", "label"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.anno_out)), exist_ok=True)
    anno.to_csv(args.anno_out, index=False)
    print(f"full annotation: {applied} positive rows relabelled, {added} reviewed pairs added, {removed} pairs removed, {before - len(anno)} merged; "
          f"wrote {args.anno_out} ({len(anno)} rows)")

    if args.pmid_csv:
        if not args.g2p_csv:
            raise SystemExit("--pmid_csv needs --g2p_csv to render the corrected thread")
        panel = build_lgmde_map(args.g2p_csv)
        rep = pd.read_csv(args.pmid_csv, dtype={"pmid": str})
        rep["_id"] = rep["g2p_lgmde"].str.split(" - ", n=1).str[0].str.strip()
        n = 0
        for c in corr.itertuples(index=False):
            if str(c.label) == "remove":
                m = (rep["_id"] == c.g2p_id_from) & ((c.pmid == "*") | (rep["pmid"] == c.pmid))
                n += int(m.sum())
                rep = rep[~m]
                continue
            if not c.g2p_id_from:
                if c.g2p_id_to not in panel:
                    raise SystemExit(f"[ERROR] {c.g2p_id_to} not in {args.g2p_csv}")
                if ((rep["pmid"] == c.pmid) & (rep["_id"] == c.g2p_id_to)).any():
                    continue
                rep = pd.concat([rep, pd.DataFrame([{"pmid": c.pmid, "g2p_lgmde": panel[c.g2p_id_to],
                                                      "label": int(c.label), "_id": c.g2p_id_to}])],
                                ignore_index=True)
                n += 1
                continue
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
