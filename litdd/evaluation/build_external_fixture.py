#!/usr/bin/env python3
"""Build an evaluation fixture from the external curated truth sets (premined/HPOA/ClinGen).

``revision/external_recall/external_positives.csv`` holds curated (pmid, g2p_id) pairs with
their TIABs. The held-out split was never used to train any component, so the full pipeline
can be run over it end to end; every abstract is a curated positive, so the meaningful
quantities are per-stage recall and the extra-entry rate (curation here is per source and
incomplete, so extras are reported, not condemned).

Writes the standard fixture contract (shards/annotated_test.parquet, gold.csv,
pairs_full.csv, provenance.json). Gold ids are resolved against ``--g2p_csv``; pairs whose
entry is retired are dropped and counted. ``--screen_preds`` (csv: pmid,...,pred) fills
``bert_predict``; without it every abstract is marked screen-positive so the fixture can be
built before the screen has run.

    python litdd/evaluation/build_external_fixture.py \\
        --external_csv revision/external_recall/external_positives.csv --split heldout \\
        --g2p_csv revision/G2P_DD_2026-06-24.csv --out_dir revision/llm_eval/external_2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from litdd.threads import build_lgmde_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--external_csv", required=True)
    ap.add_argument("--split", default="heldout", help="train / heldout / mixed / all")
    ap.add_argument("--g2p_csv", required=True)
    ap.add_argument("--screen_preds", default=None, help="csv with pmid + pred columns")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    e = pd.read_csv(args.external_csv, dtype={"pmid": str})
    if args.split != "all":
        e = e[e["split"] == args.split]
    panel = build_lgmde_map(args.g2p_csv)
    retired = sorted(set(e.loc[~e["g2p_id"].isin(panel), "g2p_id"]))
    n_retired_pairs = int((~e["g2p_id"].isin(panel)).sum())
    e = e[e["g2p_id"].isin(panel)]

    screen = None
    if args.screen_preds:
        sp = pd.read_csv(args.screen_preds, dtype={"pmid": str})
        screen = dict(zip(sp["pmid"].astype(str), sp["pred"].astype(int)))

    rows, gold_rows, pair_rows = [], [], []
    for pmid, grp in e.groupby("pmid", sort=True):
        row_id = f"pmid{pmid}"
        gold_ids = sorted(set(grp["g2p_id"]))
        bert = int(screen.get(pmid, 0)) if screen is not None else 1
        rows.append({"pmid": pmid, "row_id": row_id, "tiab": grp["tiab"].iloc[0],
                     "bert_predict": bert,
                     # bert_predict_vllm's corpus eligibility filter needs these columns;
                     # the curated sets are English-language papers and the filter is a
                     # pre-screen, not a model feature, so pass-through values are correct.
                     "languages": "eng", "pubdate": 9999})
        gold_rows.append({"pmid": pmid, "row_id": row_id,
                          "true_g2p_ids": ";".join(gold_ids), "n_gold": len(gold_ids),
                          "n_labelled_pairs": len(grp), "genereviews": False,
                          "bert_predict": bert, "gold_in_candidates": None,
                          "max_cross_score": float("nan"), "n_pmids_for_tiab": 1,
                          "sources": ";".join(sorted(set(grp["source"])))})
        for g in gold_ids:
            pair_rows.append({"pmid": pmid, "row_id": row_id, "g2p_id": g, "label": 1,
                              "in_candidates": None, "in_panel": True})

    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(args.out_dir, "shards", "annotated_test.parquet"),
                                  index=False)
    gold = pd.DataFrame(gold_rows)
    gold.to_csv(os.path.join(args.out_dir, "gold.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(args.out_dir, "pairs_full.csv"), index=False)
    prov = {"external_csv": os.path.abspath(args.external_csv), "split": args.split,
            "g2p_csv": os.path.abspath(args.g2p_csv),
            "screen_preds": os.path.abspath(args.screen_preds) if args.screen_preds else None,
            "n_abstracts": len(rows), "n_pairs": int(gold["n_gold"].sum()),
            "n_multi_entry_abstracts": int((gold["n_gold"] > 1).sum()),
            "n_screen_positive": int((gold["bert_predict"] == 1).sum()),
            "retired_pairs_dropped": n_retired_pairs, "retired_ids": retired}
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
