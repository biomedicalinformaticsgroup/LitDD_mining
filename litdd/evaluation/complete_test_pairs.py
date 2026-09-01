#!/usr/bin/env python3
"""Restore every clinician-labelled pair for the test abstracts from the full annotation file.

``final_traintest_dataset.py`` builds ``ds_test`` with ``reduce_group``: for an abstract with
any positive pair it keeps ONLY the positive pairs, for an all-negative abstract only one
negative. The clinicians, however, labelled each abstract against every DDG2P entry of its
gene(s) (an FGFR3 abstract carries labels for all eight FGFR3 entries). Those dropped sibling
negatives -- 412 pairs on 224 positive test abstracts -- are exactly what an allelic-series
evaluation needs, and they were never used in training (the split is by abstract), so
restoring them from the annotation file is completing the test labels, not re-annotating.

Writes <fixture>/pairs_full.csv with the same columns as pairs.csv (pmid, row_id, g2p_id,
label, in_panel) and checks that the positive set is unchanged from gold.csv.

    python litdd/evaluation/complete_test_pairs.py \\
        --anno_csv $REF/g2p_id_tiab_anno_df_FINAL.csv \\
        --fixture_dir revision/llm_eval/annotated_2026 --g2p_csv revision/G2P_DD_2026-06-24.csv
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
    ap.add_argument("--anno_csv", required=True, help="g2p_id, pmid, tiab, label -- every labelled pair")
    ap.add_argument("--fixture_dir", required=True, help="dir with shards/annotated_test.parquet and gold.csv")
    ap.add_argument("--g2p_csv", required=True, help="the export this fixture uses (in_panel flag)")
    args = ap.parse_args()

    shard = pd.read_parquet(os.path.join(args.fixture_dir, "shards", "annotated_test.parquet"),
                            columns=["pmid", "row_id", "tiab"])
    gold = pd.read_csv(os.path.join(args.fixture_dir, "gold.csv"))
    gold["row_id"] = gold["row_id"].astype(str)
    anno = pd.read_csv(args.anno_csv, usecols=["g2p_id", "pmid", "tiab", "label"])
    panel = build_lgmde_map(args.g2p_csv)

    t2row = dict(zip(shard["tiab"], shard["row_id"].astype(str)))
    t2pmid = dict(zip(shard["tiab"], shard["pmid"].astype(str)))
    a = anno[anno["tiab"].isin(t2row)].copy()
    a["row_id"] = a["tiab"].map(t2row)
    a["pmid"] = a["tiab"].map(t2pmid)
    a["label"] = a["label"].astype(int)
    a = (a.groupby(["row_id", "g2p_id"], as_index=False)
         .agg(pmid=("pmid", "first"), label=("label", "max")))   # a pair labelled twice: positive wins
    a["in_panel"] = a["g2p_id"].isin(panel)

    # the positive set must be exactly gold.csv (reduce_group kept every positive)
    pos = a[(a["label"] == 1) & a["in_panel"]].groupby("row_id")["g2p_id"].agg(lambda s: ";".join(sorted(set(s))))
    g = gold.set_index("row_id")["true_g2p_ids"].fillna("")
    diff = [(rid, g.get(rid, ""), pos.get(rid, "")) for rid in set(g.index) | set(pos.index)
            if (g.get(rid, "") or "") != (pos.get(rid, "") or "")]
    if diff:
        raise SystemExit(f"[ERROR] positive set differs from gold.csv for {len(diff)} abstracts, "
                         f"e.g. {diff[:3]}")

    old = pd.read_csv(os.path.join(args.fixture_dir, "pairs.csv"))
    out = a[["pmid", "row_id", "g2p_id", "label", "in_panel"]].sort_values(["row_id", "g2p_id"])
    out.to_csv(os.path.join(args.fixture_dir, "pairs_full.csv"), index=False)
    pos_rows = set(out.loc[out["label"] == 1, "row_id"])
    report = {
        "pairs_in_split": int(len(old)), "pairs_full": int(len(out)),
        "positives": int((out["label"] == 1).sum()), "negatives": int((out["label"] == 0).sum()),
        "negatives_on_positive_abstracts": int(((out["label"] == 0) & out["row_id"].isin(pos_rows)).sum()),
        "positive_abstracts_with_negatives": int(out[(out["label"] == 0) & out["row_id"].isin(pos_rows)]["row_id"].nunique()),
        "pairs_not_in_panel": int((~out["in_panel"]).sum()),
        "gold_unchanged": True,
    }
    with open(os.path.join(args.fixture_dir, "pairs_full.provenance.json"), "w") as f:
        json.dump({"anno_csv": os.path.abspath(args.anno_csv), **report}, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
