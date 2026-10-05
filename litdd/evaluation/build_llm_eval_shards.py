#!/usr/bin/env python3
"""Build the annotated-set fixture for evaluating the adjudication stage.

The annotated split is pair-level: each row is a (tiab, g2p_lgmde, label) pair and an
abstract can carry several positive entries and several labelled negatives. This script
turns a saved split into the fixture contract:

  <out_dir>/shards/annotated_test.parquet   one row per abstract: pmid, row_id, tiab, bert_predict
  <out_dir>/gold.csv                        pmid, row_id, true_g2p_ids (';'-joined), n_gold,
                                            genereviews, bert_predict, n_pmids_for_tiab
  <out_dir>/pairs.csv                       every labelled (pmid, row_id, g2p_id, label, in_panel) pair
  <out_dir>/provenance.json

Candidates are not part of the fixture: run ``litdd.pipeline.gene_candidates`` on the shard
parquet and ``litdd.pipeline.build_llm_shards`` on its output to produce the adjudication
input, then evaluate with ``llm_adjudication_eval.py`` against ``gold.csv``.

``--g2p_csv`` must be the export the evaluation uses: labels are resolved by G2P id, and with
``--drop_retired`` abstracts whose gold entry is absent from the export are dropped and
counted; otherwise a retired gold id aborts. ``--screen_preds`` supplies per-PMID screen
predictions (csv: pmid, pred) for ``bert_predict``; without it every abstract is marked
screen-positive. ``--corrections`` applies ``data/annotation_corrections.csv`` before the
gold and pairs files are written.

    python -m litdd.evaluation.build_llm_eval_shards --dataset_dir data/ds_test \\
        --anno_csv g2p_id_tiab_anno_df_FINAL.csv --g2p_csv G2P_DD.csv \\
        --corrections data/annotation_corrections.csv --screen_preds screen_preds.csv \\
        --out_dir fixture/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import time

import pandas as pd

from litdd.threads import build_lgmde_map

logger = logging.getLogger(__name__)
G2P_ID_RE = re.compile(r"^(G2P\d+)")
GENEREVIEWS_MARKER = "CLINICAL CHARACTERISTICS"


def thread_id(thread: str) -> str:
    m = G2P_ID_RE.match(str(thread).strip())
    if not m:
        raise ValueError(f"thread does not start with a G2P id: {thread[:80]!r}")
    return m.group(1)


def verify_panel(df: pd.DataFrame, panel: dict, g2p_csv: str) -> dict:
    """Report which gold ids exist in the export."""
    gold_ids = {thread_id(g) for g in df.loc[df["label"] == 1, "g2p_lgmde"].unique()}
    return {"g2p_csv": g2p_csv, "panel_entries": len(panel), "gold_ids": len(gold_ids),
            "gold_ids_in_panel": sum(i in panel for i in gold_ids),
            "gold_ids_retired": sorted(i for i in gold_ids if i not in panel)}


def build(df: pd.DataFrame, anno: pd.DataFrame, panel: dict, drop_retired: bool,
          screen_preds: dict | None):
    """Shard, gold, pairs and dropped frames from the pair-level split."""
    df = df.copy()
    df["g2p_id"] = df["g2p_lgmde"].map(thread_id)
    t2p = (anno.dropna(subset=["pmid"]).assign(pmid=lambda d: d["pmid"].astype(int).astype(str))
           .groupby("tiab")["pmid"].agg(lambda s: sorted(set(s))))
    rows, gold_rows, pair_rows, dropped = [], [], [], []
    for tiab, grp in df.groupby("tiab", sort=False):
        pmids = t2p.get(tiab, [])
        pmid = pmids[0] if pmids else None
        gold_ids = sorted(set(grp.loc[grp["label"] == 1, "g2p_id"]))
        row_id = f"pmid{pmid}" if pmid else f"tiab{len(rows) + len(dropped):05d}"
        retired = [g for g in gold_ids if g not in panel]
        if retired:
            if not drop_retired:
                raise SystemExit(f"gold id(s) {retired} for {row_id} are not in the export; "
                                 "pass --drop_retired or use the matching export")
            dropped.append({"row_id": row_id, "pmid": pmid, "retired_gold_ids": ";".join(retired)})
            continue
        bert = int(grp["bert_predict"].iloc[0]) if "bert_predict" in grp.columns else 1
        if screen_preds is not None:
            if pmid is None or pmid not in screen_preds:
                raise SystemExit(f"--screen_preds has no prediction for pmid {pmid}")
            bert = int(screen_preds[pmid])
        rows.append({"pmid": pmid or row_id, "row_id": row_id, "tiab": tiab, "bert_predict": bert})
        gold_rows.append({
            "pmid": pmid or row_id, "row_id": row_id,
            "true_g2p_ids": ";".join(gold_ids), "n_gold": len(gold_ids),
            "n_labelled_pairs": len(grp),
            "genereviews": GENEREVIEWS_MARKER in tiab,
            "bert_predict": bert,
            "n_pmids_for_tiab": len(pmids),
        })
        for _, r in grp.iterrows():
            pair_rows.append({"pmid": pmid or row_id, "row_id": row_id,
                              "g2p_id": r["g2p_id"], "label": int(r["label"]),
                              "in_panel": r["g2p_id"] in panel})
    return pd.DataFrame(rows), pd.DataFrame(gold_rows), pd.DataFrame(pair_rows), pd.DataFrame(dropped)


def apply_corrections(df: pd.DataFrame, anno: pd.DataFrame, corrections_csv: str, panel: dict,
                      g2p_csv: str) -> tuple[pd.DataFrame, int]:
    """Apply relabel, add and remove rows from the corrections CSV to the pair frame."""
    corr = pd.read_csv(corrections_csv, dtype=str)
    pm = (anno.dropna(subset=["pmid"]).assign(pmid=lambda d: d["pmid"].astype(int).astype(str))
          .drop_duplicates("tiab").set_index("tiab")["pmid"])
    df = df.copy()
    df["_pmid"] = df["tiab"].map(pm)
    df["_id"] = df["g2p_lgmde"].map(thread_id)
    corr["g2p_id_from"] = corr["g2p_id_from"].fillna("")
    if "label" not in corr.columns:
        corr["label"] = "1"
    n_corr = 0
    for c in corr[corr["label"].astype(str) == "remove"].itertuples(index=False):
        m = (df["_id"] == c.g2p_id_from) & ((c.pmid == "*") | (df["_pmid"] == c.pmid))
        n_corr += int(m.sum())
        df.loc[m, "label"] = 0
    corr = corr[corr["label"].astype(str) != "remove"]
    add_rows = []
    for c in corr.itertuples(index=False):
        if c.g2p_id_to not in panel:
            if c.g2p_id_from or int(c.label) == 1:
                logger.warning("correction target %s not in %s; skipped", c.g2p_id_to, g2p_csv)
            continue
        if c.g2p_id_from:
            m = (df["_pmid"] == c.pmid) & (df["_id"] == c.g2p_id_from) & (df["label"] == 1)
            if m.any():
                df.loc[m, "g2p_lgmde"] = panel[c.g2p_id_to]
                n_corr += int(m.sum())
        elif int(c.label) == 1:
            src = df[df["_pmid"] == c.pmid]
            if len(src) and not ((df["_pmid"] == c.pmid) & (df["_id"] == c.g2p_id_to)).any():
                row = src.iloc[0].copy()
                row["g2p_lgmde"] = panel[c.g2p_id_to]
                row["label"] = 1
                add_rows.append(row)
                n_corr += 1
    if add_rows:
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
    return df.drop(columns=["_pmid", "_id"]), n_corr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset_dir", required=True,
                    help="a save_to_disk dataset with tiab, g2p_lgmde, label")
    ap.add_argument("--anno_csv", required=True, help="pair annotation CSV with pmid and tiab columns")
    ap.add_argument("--g2p_csv", required=True, help="the export this evaluation uses")
    ap.add_argument("--drop_retired", action="store_true",
                    help="drop abstracts whose gold entry is not in --g2p_csv and record them")
    ap.add_argument("--screen_preds", default=None, help="csv (pmid, pred) with the screen's predictions")
    ap.add_argument("--corrections", default=None, help="data/annotation_corrections.csv")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    from datasets import load_from_disk
    df = load_from_disk(args.dataset_dir).to_pandas()
    df["label"] = df["label"].astype(int)
    source = os.path.abspath(args.dataset_dir)
    need = {"tiab", "g2p_lgmde", "label"}
    if not need <= set(df.columns):
        raise SystemExit(f"{source} lacks {need - set(df.columns)}")
    anno = pd.read_csv(args.anno_csv, usecols=["pmid", "tiab"])
    panel = build_lgmde_map(args.g2p_csv)
    n_corr = 0
    if args.corrections:
        df, n_corr = apply_corrections(df, anno, args.corrections, panel, args.g2p_csv)
        logger.info("applied %d label corrections from %s", n_corr, args.corrections)
    screen = None
    if args.screen_preds:
        sp = pd.read_csv(args.screen_preds, dtype={"pmid": str})
        screen = dict(zip(sp["pmid"].astype(str), sp["pred"].astype(int)))

    panel_report = verify_panel(df, panel, args.g2p_csv)
    shard, gold, pairs, dropped = build(df, anno, panel, args.drop_retired, screen)

    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)
    shard_path = os.path.join(args.out_dir, "shards", "annotated_test.parquet")
    shard.to_parquet(shard_path, index=False)
    gold.to_csv(os.path.join(args.out_dir, "gold.csv"), index=False)
    pairs.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    if len(dropped):
        dropped.to_csv(os.path.join(args.out_dir, "dropped_retired.csv"), index=False)

    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip() or None
    except OSError:
        sha = None
    prov = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "litdd/evaluation/build_llm_eval_shards.py", "git_commit": sha,
        "source": source,
        "anno_csv": os.path.abspath(args.anno_csv),
        "corrections": os.path.abspath(args.corrections) if args.corrections else None,
        "n_label_corrections_applied": n_corr,
        "screen_preds": os.path.abspath(args.screen_preds) if args.screen_preds else None,
        "panel_check": panel_report,
        "n_pairs": int(len(df)), "n_tiabs": int(len(shard)),
        "n_tiabs_dropped_retired_gold": int(len(dropped)),
        "n_tiabs_with_gold": int((gold["n_gold"] > 0).sum()),
        "n_tiabs_multi_gold": int((gold["n_gold"] > 1).sum()),
        "n_gold_ids": int(gold["n_gold"].sum()),
        "n_screen_positive": int((gold["bert_predict"] == 1).sum()),
        "n_screen_positive_with_gold": int(((gold["bert_predict"] == 1) & (gold["n_gold"] > 0)).sum()),
        "n_genereviews": int(gold["genereviews"].sum()),
        "n_tiabs_without_pmid": int(gold["row_id"].str.startswith("tiab").sum()),
        "n_tiabs_multi_pmid": int((gold["n_pmids_for_tiab"] > 1).sum()),
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
    logger.info("wrote %s (%d rows)", shard_path, len(shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
